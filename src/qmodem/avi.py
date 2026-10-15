"""Adversarial variational inference (AVI): a dataset-agnostic inference method.

An AVI method's approximate posterior over the inferred subnetwork is implicit: a
**generator** (a latent map followed by a postprocessor) turns one random scalar into
one draw of the inferred parameters. The generator is fitted adversarially against a
**discriminator** that judges input-target pairs:

```
generator   w_s = G_θ(z_s),  z_s ~ U(0, 2π),  s = 1..S       (latent map -> postprocessor)
fakes       ŷ_s = μ_{w_s}(x) + √σ²_ale(x)·ε                  (σ²_ale = problem.aleatoric_variance)
disc ×k     L_D = −mean_B log D(x, y) − mean_{S,B} log(1 − D(x, ŷ_s))
gen  ×1     L_G = mean_S[−loglik_{N/B}(w_s) − log p(w_s)] + λ·N·mean_{S,B}[−log D(x, ŷ_s)]
```

``AVI`` satisfies stagereg's ``InferenceMethod`` seam. It composes three components,
built from factories once the posterior problem is known: the generator, which
contains a ``LatentMap`` (quantum for Q-AVI, classical for C-AVI), and the
discriminator. Nothing here knows the network, the data or their shapes; the battery
generator and discriminator live with the battery models.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax
import pennylane as qp
from flax import nnx
from jax.flatten_util import ravel_pytree
from stagereg.core import Batch
from stagereg.jax import PosteriorProblem
from stagereg.jax.posterior import draw_unlimited, require_inferred_tree
from stagereg.jax.rng import network_key

from qmodem.quantum_circuits import ContinuousCircuitFactory

type PyTree = Any

LATENT_UPPER = 2 * math.pi
"""Latent inputs are drawn from ``U(0, LATENT_UPPER)``."""

WEIGHT_SPREAD_DRAWS = 8
"""Draws per update used for the ``weight_spread`` diagnostic.

A fixed count, independent of the training draws S (default 1, for which a spread across
the training draws would always be zero).
"""

DISCRIMINATOR_LOSS = "discriminator_loss"
DISCRIMINATOR_ACCURACY = "discriminator_accuracy"
WEIGHT_SPREAD = "weight_spread"


@runtime_checkable
class LatentMap(Protocol):
    """Maps one random scalar to a bounded feature vector of width ``q``.

    A latent map is the first part of an AVI generator. Implementations are
    ``nnx.Module`` s whose trainable parameters are ``nnx.Param`` s.
    """

    @property
    def width(self) -> int:
        """The width ``q`` of the latent vector."""
        ...

    @property
    def parameter_count(self) -> int:
        """The number of trainable parameters of the latent map."""
        ...

    def __call__(self, z: jax.Array) -> jax.Array:
        """Map a scalar to the latent vector.

        Args:
            z: A scalar random input.

        Returns:
            The latent vector of shape ``(q,)``, with entries in ``[-1, 1]``.
        """
        ...


class Generator(Protocol):
    """Maps one random scalar to one draw of the inferred parameters.

    Implementations are ``nnx.Module`` s whose ``nnx.Param`` s are trained; other
    state (e.g. a fixed offset) is held constant.
    """

    def __call__(self, z: jax.Array) -> PyTree:
        """Generate one draw.

        Args:
            z: A scalar random input from ``U(0, LATENT_UPPER)``.

        Returns:
            Parameters with the inferred subnetwork's tree.
        """
        ...


class Discriminator(Protocol):
    """Judges whether an input-target pair is real (an ``nnx.Module``)."""

    def __call__(self, inputs: PyTree, targets: jax.Array) -> jax.Array:
        """Score a batch of pairs.

        Args:
            inputs: Model inputs with a leading batch dimension ``B``.
            targets: Targets (real or generated) shaped like the mean predictions.

        Returns:
            Logits of ``P(real)``, shape ``(B, 1)``.
        """
        ...


type GeneratorFactory = Callable[[PyTree, jax.Array], Generator]
"""Builds a generator from the warm start (the inferred parameters) and a key."""

type DiscriminatorFactory = Callable[[jax.Array], Discriminator]
"""Builds a discriminator from a key."""


class QuantumLatentMap(nnx.Module):
    """The quantum latent map ``z ↦ ⟨Z⟩ ∈ [-1, 1]^q`` of Q-AVI.

    A ``q``-qubit, ``L``-layer parametrized circuit (``ContinuousCircuitFactory``):
    ``RY(z)`` on every qubit, then ``L`` layers of ``RY``/``RZ`` rotations and a CNOT
    ring, measured as the Pauli-Z expectation of every qubit. It has ``2Lq``
    parameters, initialized ``U(0, 2π)``.
    """

    def __init__(self, n_qubits: int, n_layers: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the circuit and its angles.

        Args:
            n_qubits: Number of qubits ``q`` (the latent width).
            n_layers: Number of variational layers ``L``.
            rngs: RNGs of the angle initializer.

        Raises:
            ValueError: If there are fewer than two qubits (the CNOT ring needs two).
        """
        if n_qubits < 2:
            raise ValueError("a quantum latent map needs at least two qubits.")
        factory = ContinuousCircuitFactory(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = factory.make_circuit(qp.device("default.qubit", wires=n_qubits))
        self.angles = nnx.Param(
            jax.random.uniform(
                rngs.params(), (n_layers, n_qubits, 2), maxval=LATENT_UPPER
            )
        )

    @property
    def width(self) -> int:
        """The latent width ``q`` (the number of qubits)."""
        return self.n_qubits

    @property
    def parameter_count(self) -> int:
        """``2Lq``: one ``RY`` and one ``RZ`` angle per qubit and layer."""
        return 2 * self.n_layers * self.n_qubits

    def __call__(self, z: jax.Array) -> jax.Array:
        """Evaluate the Pauli-Z expectations after encoding ``z``.

        Args:
            z: A scalar random input.

        Returns:
            The expectations, shape ``(q,)``.
        """
        return jnp.stack(self.circuit(z, self.angles[...]))


# --- posterior ---------------------------------------------------------------------


def _latent_input(key: jax.Array) -> jax.Array:
    """Draw one latent scalar ``z ~ U(0, LATENT_UPPER)``."""
    return jax.random.uniform(key, (), maxval=LATENT_UPPER)


@dataclass(frozen=True)
class _GeneratorModel:
    """A generator as a pure function of its trainable parameters."""

    graphdef: nnx.GraphDef
    constants: nnx.State

    def __call__(self, parameters: nnx.State, z: jax.Array) -> PyTree:
        """Generate one draw with ``parameters``.

        Args:
            parameters: The generator's ``nnx.Param`` state.
            z: A scalar random input.

        Returns:
            One draw of the inferred parameters.
        """
        return nnx.merge(self.graphdef, parameters, self.constants)(z)

    def draw(self, parameters: nnx.State, key: jax.Array) -> PyTree:
        """Generate the draw selected by ``key``.

        Args:
            parameters: The generator's ``nnx.Param`` state.
            key: The draw's key.

        Returns:
            One draw of the inferred parameters.
        """
        return self(parameters, _latent_input(key))


@dataclass(frozen=True)
class AVIPosterior:
    """The implicit posterior of an AVI method.

    Draw ``s`` under ``key`` feeds the generator ``z ~ U(0, 2π)`` from
    ``draw_key = fold_in(key, s)`` and evaluates the mean function with
    ``network_key(draw_key)``, so it is the same mean function at every input and in
    every call.

    Attributes:
        mean_function: The problem's ``(parameters, inputs, key) -> means``.
        generator: The generator as a function of its parameters.
        parameters: The generator's trained parameters.
    """

    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    generator: _GeneratorModel
    parameters: nnx.State
    _sample: Callable[..., jax.Array] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile the draw function once per posterior."""

        def sample(
            parameters: nnx.State, inputs: PyTree, key: jax.Array, num_samples: int
        ) -> jax.Array:
            def draw(index: jax.Array) -> jax.Array:
                draw_key = jax.random.fold_in(key, index)
                weights = self.generator.draw(parameters, draw_key)
                return self.mean_function(weights, inputs, network_key(draw_key))

            return draw_unlimited(draw, num_samples)

        object.__setattr__(self, "_sample", jax.jit(sample, static_argnums=3))

    @property
    def num_draws(self) -> int | None:
        """Always ``None``: a generator has unlimited draws."""
        return None

    def sample_means(
        self, inputs: PyTree, key: jax.Array, num_samples: int | None = None
    ) -> jax.Array:
        """Draw mean predictions at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key selecting the draws.
            num_samples: The number of draws ``S``, positive.

        Returns:
            The draws' mean predictions ``[S, *batch]``.

        Raises:
            ValueError: If ``num_samples`` is omitted or not positive.
        """
        if num_samples is None or num_samples <= 0:
            raise ValueError("an AVI posterior needs a positive num_samples.")
        return self._sample(self.parameters, inputs, key, num_samples)


# --- inference method ----------------------------------------------------------------


@dataclass(frozen=True)
class _Components:
    """The live AVI components after ``init``."""

    problem: PosteriorProblem
    generator: _GeneratorModel
    discriminator_graphdef: nnx.GraphDef
    discriminator_constants: nnx.State
    step: Callable[..., tuple[PyTree, Mapping[str, jax.Array]]]


@dataclass
class AVI:
    """Adversarial variational inference over the inferred subnetwork.

    Every update takes ``discriminator_steps`` (k) discriminator steps and one
    generator step on the same batch, each with ``num_draws`` (S) whole-function
    draws. The generator's objective honours the posterior problem's likelihood and
    prior and adds ``adversarial_weight · N`` times the mean adversarial error. The
    reported ``loss`` is the generator loss before its step.

    It supports early stopping. ``state()`` holds the generator, the discriminator
    and both optimizer states; ``posterior_state()`` holds the generator only, so the
    discriminator never leaves training.

    Attributes:
        build_generator: Builds the generator from the warm start and a key.
        build_discriminator: Builds the discriminator from a key.
        key: Initializes the generator (``fold_in(key, 0)``) and the
            discriminator (``fold_in(key, 1)``).
        generator_optimizer: Steps the generator's parameters.
        discriminator_optimizer: Steps the discriminator's parameters.
        adversarial_weight: The adversarial weight λ, per example.
        num_draws: Whole-function draws S per step.
        discriminator_steps: Discriminator steps k per generator step.
    """

    build_generator: GeneratorFactory
    build_discriminator: DiscriminatorFactory
    key: jax.Array
    generator_optimizer: optax.GradientTransformation
    discriminator_optimizer: optax.GradientTransformation
    adversarial_weight: float
    num_draws: int = 1
    discriminator_steps: int = 1
    _components: _Components | None = field(default=None, init=False, repr=False)
    _state: PyTree = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If λ is negative or not finite, or S or k is not positive.
        """
        if not math.isfinite(self.adversarial_weight) or self.adversarial_weight < 0:
            raise ValueError("adversarial_weight must be finite and non-negative.")
        if self.num_draws < 1 or self.discriminator_steps < 1:
            raise ValueError("num_draws and discriminator_steps must be positive.")

    @property
    def supports_early_stopping(self) -> bool:
        """Always ``True``: the generator can stop at any epoch."""
        return True

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Discriminator loss and accuracy, and the weight spread across draws."""
        return (DISCRIMINATOR_LOSS, DISCRIMINATOR_ACCURACY, WEIGHT_SPREAD)

    @property
    def has_posterior(self) -> bool:
        """Whether the method has been initialized."""
        return self._components is not None

    def init(self, problem: PosteriorProblem) -> None:
        """Build the generator and discriminator and their optimizers.

        Args:
            problem: The posterior problem.
        """
        generator = self.build_generator(
            problem.initial_parameters, jax.random.fold_in(self.key, 0)
        )
        discriminator = self.build_discriminator(jax.random.fold_in(self.key, 1))
        generator_graphdef, generator_parameters, generator_constants = nnx.split(
            generator, nnx.Param, ...
        )
        (
            discriminator_graphdef,
            discriminator_parameters,
            discriminator_constants,
        ) = nnx.split(discriminator, nnx.Param, ...)
        generator_model = _GeneratorModel(generator_graphdef, generator_constants)
        self._components = _Components(
            problem=problem,
            generator=generator_model,
            discriminator_graphdef=discriminator_graphdef,
            discriminator_constants=discriminator_constants,
            step=_make_step(
                problem=problem,
                generator=generator_model,
                discriminator=_DiscriminatorModel(
                    discriminator_graphdef, discriminator_constants
                ),
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                adversarial_weight=self.adversarial_weight,
                num_draws=self.num_draws,
                discriminator_steps=self.discriminator_steps,
            ),
        )
        self._state = {
            "generator": generator_parameters,
            "discriminator": discriminator_parameters,
            "generator_optimizer": self.generator_optimizer.init(generator_parameters),
            "discriminator_optimizer": self.discriminator_optimizer.init(
                discriminator_parameters
            ),
        }

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Take k discriminator steps and one generator step on ``batch``.

        Args:
            batch: The training batch.
            key: A fresh key for this batch's draws and noise.

        Returns:
            ``{"loss": generator_loss, **diagnostics}``.

        Raises:
            ValueError: If the method has not been initialized.
        """
        components = self._require_components()
        self._state, outputs = components.step(
            self._state, batch.inputs, batch.targets, batch.sample_weight, key
        )
        return outputs

    def posterior(self) -> AVIPosterior:
        """Return the current generator's implicit posterior.

        Returns:
            A posterior independent of later updates.

        Raises:
            ValueError: If the method has not been initialized.
        """
        components = self._require_components()
        return AVIPosterior(
            mean_function=components.problem.mean_function,
            generator=components.generator,
            parameters=self._state["generator"],
        )

    def state(self) -> PyTree:
        """Return the generator, the discriminator and both optimizer states.

        Returns:
            ``{"generator", "discriminator", "generator_optimizer",
            "discriminator_optimizer"}``.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_components()
        return dict(self._state)

    def load_state(self, state: PyTree) -> None:
        """Resume inference from a state returned by ``state()``.

        Args:
            state: A state from ``state`` after ``init`` on the same problem.

        Raises:
            ValueError: If the method has not been initialized or a part's tree
                differs from this method's.
        """
        self._require_components()
        for name, current in self._state.items():
            _require_same_tree(state[name], current, name)
        self._state = dict(state)

    def posterior_state(self) -> PyTree:
        """Return the generator's parameters (the discriminator is left out).

        Returns:
            ``{"generator": parameters}``.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_components()
        return {"generator": self._state["generator"]}

    def load_posterior(self, state: PyTree) -> None:
        """Rebuild the posterior from a state returned by ``posterior_state()``.

        Args:
            state: A state from ``posterior_state``.

        Raises:
            ValueError: If the method has not been initialized or the generator's
                tree differs from this method's (another generator, or a posterior
                of another inferred subnetwork).
        """
        components = self._require_components()
        _require_same_tree(state["generator"], self._state["generator"], "generator")
        require_inferred_tree(
            components.generator.draw(state["generator"], jax.random.key(0)),
            components.problem.initial_parameters,
        )
        self._state = {**self._state, "generator": state["generator"]}

    def _require_components(self) -> _Components:
        """Return the live components, refusing an uninitialized method."""
        if self._components is None:
            raise ValueError("AVI must be initialized with init().")
        return self._components


def _require_same_tree(actual: PyTree, expected: PyTree, name: str) -> None:
    """Refuse a state part whose tree is not the live one's.

    Raises:
        ValueError: If the trees' structures differ.
    """
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise ValueError(f"the {name} state's tree differs from this AVI method's.")


# --- loss --------------------------------------------------------------------------


@dataclass(frozen=True)
class _DiscriminatorModel:
    """A discriminator as a pure function of its trainable parameters."""

    graphdef: nnx.GraphDef
    constants: nnx.State

    def __call__(
        self, parameters: nnx.State, inputs: PyTree, targets: jax.Array
    ) -> jax.Array:
        """Score pairs with ``parameters``.

        Args:
            parameters: The discriminator's ``nnx.Param`` state.
            inputs: Model inputs with a leading batch dimension.
            targets: Targets shaped like the mean predictions.

        Returns:
            Logits of ``P(real)``, shape ``(B, 1)``.
        """
        return nnx.merge(self.graphdef, parameters, self.constants)(inputs, targets)


def discriminator_loss(
    real_logits: jax.Array, fake_logits: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Equally weighted BCE of real against generated pairs, and its accuracy.

    ``L_D = −mean_B log D(x, y) − mean_{S,B} log(1 − D(x, ŷ_s))``.

    Args:
        real_logits: Logits of the real pairs, ``(B, 1)``.
        fake_logits: Logits of the generated pairs, ``(S, B, 1)``.

    Returns:
        The loss and the accuracy at threshold 1/2 (mean of the real and the
        fake accuracies).
    """
    loss = jnp.mean(jax.nn.softplus(-real_logits)) + jnp.mean(
        jax.nn.softplus(fake_logits)
    )
    accuracy = 0.5 * (jnp.mean(real_logits > 0) + jnp.mean(fake_logits <= 0))
    return loss, accuracy


def generator_loss(
    *,
    log_likelihoods: jax.Array,
    log_priors: jax.Array,
    fake_logits: jax.Array,
    adversarial_weight: float,
    dataset_size: int,
) -> jax.Array:
    """The AVI generator objective.

    ``L_G = mean_S[−loglik_{N/B}(w_s) − log p(w_s)] + λ·N·mean_{S,B}[−log D(x, ŷ_s)]``.

    Args:
        log_likelihoods: The minibatch log-likelihood of each draw (already scaled
            by ``N / B``), ``(S,)``.
        log_priors: The prior log-density of each draw, ``(S,)``.
        fake_logits: Discriminator logits of the generated pairs, ``(S, B, 1)``.
        adversarial_weight: The adversarial weight λ.
        dataset_size: The dataset size N.

    Returns:
        The scalar generator loss.
    """
    adversarial = jnp.mean(jax.nn.softplus(-fake_logits))
    return (
        jnp.mean(-log_likelihoods - log_priors)
        + adversarial_weight * dataset_size * adversarial
    )


def weight_spread(draws: PyTree) -> jax.Array:
    """Mean standard deviation of the inferred parameters across draws.

    Args:
        draws: Draws of the inferred parameters stacked on a leading axis.

    Returns:
        The mean over parameters of their standard deviation across draws.
    """
    flat = jax.vmap(lambda tree: ravel_pytree(tree)[0])(draws)
    return jnp.mean(jnp.std(flat, axis=0))


def _make_step(
    *,
    problem: PosteriorProblem,
    generator: _GeneratorModel,
    discriminator: _DiscriminatorModel,
    generator_optimizer: optax.GradientTransformation,
    discriminator_optimizer: optax.GradientTransformation,
    adversarial_weight: float,
    num_draws: int,
    discriminator_steps: int,
) -> Callable[..., tuple[PyTree, Mapping[str, jax.Array]]]:
    """Return the jitted AVI update: k discriminator steps, then one generator step."""
    if problem.aleatoric_variance is None:
        raise ValueError("AVI needs the problem's aleatoric variance.")
    aleatoric_variance = problem.aleatoric_variance

    def draws_and_fakes(
        generator_parameters: nnx.State, inputs: PyTree, key: jax.Array
    ) -> tuple[PyTree, jax.Array]:
        """Draw S parameter sets and one predictive sample per draw and window."""
        standard_deviation = jnp.sqrt(aleatoric_variance(inputs))

        def one(index: jax.Array) -> tuple[PyTree, jax.Array]:
            draw_key = jax.random.fold_in(key, index)
            weights = generator.draw(generator_parameters, draw_key)
            means = problem.mean_function(weights, inputs, network_key(draw_key))
            noise = jax.random.normal(
                jax.random.fold_in(draw_key, 1), means.shape, means.dtype
            )
            return weights, means + standard_deviation * noise

        return jax.vmap(one)(jnp.arange(num_draws))

    def fake_logits(
        discriminator_parameters: nnx.State, inputs: PyTree, fakes: jax.Array
    ) -> jax.Array:
        return jax.vmap(
            lambda fake: discriminator(discriminator_parameters, inputs, fake)
        )(fakes)

    def discriminator_objective(
        discriminator_parameters: nnx.State,
        inputs: PyTree,
        targets: jax.Array,
        fakes: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        real = discriminator(discriminator_parameters, inputs, targets)
        return discriminator_loss(
            real, fake_logits(discriminator_parameters, inputs, fakes)
        )

    def generator_objective(
        generator_parameters: nnx.State,
        discriminator_parameters: nnx.State,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        weights, fakes = draws_and_fakes(generator_parameters, inputs, key)
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        likelihood_keys = jax.random.split(jax.random.fold_in(key, 2), num_draws)
        log_likelihoods = jax.vmap(
            lambda w, k: problem.log_likelihood(w, batch, network_key(k))
        )(weights, likelihood_keys)
        log_priors = jax.vmap(problem.prior.log_prob)(weights)
        return generator_loss(
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            fake_logits=fake_logits(discriminator_parameters, inputs, fakes),
            adversarial_weight=adversarial_weight,
            dataset_size=problem.dataset_size,
        )

    @jax.jit
    def step(
        state: PyTree,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> tuple[PyTree, Mapping[str, jax.Array]]:
        discriminator_keys = jax.random.split(
            jax.random.fold_in(key, 0), discriminator_steps
        )
        generator_key = jax.random.fold_in(key, 1)
        spread_key = jax.random.fold_in(key, 2)

        def discriminator_step(
            carry: tuple[nnx.State, PyTree], step_key: jax.Array
        ) -> tuple[tuple[nnx.State, PyTree], tuple[jax.Array, jax.Array]]:
            parameters, optimizer_state = carry
            _, fakes = draws_and_fakes(state["generator"], inputs, step_key)
            fakes = jax.lax.stop_gradient(fakes)
            (loss, accuracy), grads = jax.value_and_grad(
                discriminator_objective, has_aux=True
            )(parameters, inputs, targets, fakes)
            updates, optimizer_state = discriminator_optimizer.update(
                grads, optimizer_state, parameters
            )
            return (optax.apply_updates(parameters, updates), optimizer_state), (
                loss,
                accuracy,
            )

        (discriminator_parameters, discriminator_state), (d_losses, accuracies) = (
            jax.lax.scan(
                discriminator_step,
                (state["discriminator"], state["discriminator_optimizer"]),
                discriminator_keys,
            )
        )

        loss, grads = jax.value_and_grad(generator_objective)(
            state["generator"],
            discriminator_parameters,
            inputs,
            targets,
            sample_weight,
            generator_key,
        )
        updates, generator_state = generator_optimizer.update(
            grads, state["generator_optimizer"], state["generator"]
        )
        spread_draws = jax.vmap(lambda k: generator.draw(state["generator"], k))(
            jax.random.split(spread_key, WEIGHT_SPREAD_DRAWS)
        )
        new_state = {
            "generator": optax.apply_updates(state["generator"], updates),
            "discriminator": discriminator_parameters,
            "generator_optimizer": generator_state,
            "discriminator_optimizer": discriminator_state,
        }
        return new_state, {
            "loss": loss,
            DISCRIMINATOR_LOSS: d_losses[-1],
            DISCRIMINATOR_ACCURACY: accuracies[-1],
            WEIGHT_SPREAD: weight_spread(spread_draws),
        }

    return step
