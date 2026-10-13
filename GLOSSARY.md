# qmodem

The shared vocabulary of qmodem: quantum-enhanced uncertainty quantification for remaining-useful-life regression, and the protocol that compares quantum and classical inference methods. Training-run terms (stage, posterior stage, inference method, posterior, draw, aleatoric/epistemic variance) are stagereg's and are defined in the [stagereg glossary](https://github.com/GroupiSP/stagereg/blob/main/GLOSSARY.md); they are not redefined here.

## Inference methods

**QAVI**:
Quantum adversarial variational inference: the inference method whose approximate posterior over part of the network's weights is implicit, defined by a generator, and fitted adversarially against a discriminator.
_Avoid_: Quantum BNN, quantum hypernetwork

**Generator**:
A parametrized quantum circuit followed by a postprocessor, mapping a random input to one draw of the inferred weights.
_Avoid_: Weights generator, hypernetwork

**Postprocessor**:
The classical map that turns a generator's circuit measurements into weights of the inferred part of the network.
_Avoid_: Readout, decoder

**Discriminator**:
A classical network, external to the regression network, that judges whether an input–target pair is real or produced by the regression network; it exists only to train QAVI.
_Avoid_: Critic

**MCD**:
Monte Carlo dropout: the inference method that keeps dropout active at inference, so each dropout mask is one draw.
_Avoid_: MC-Dropout, dropout BNN

**BBB**:
Bayes by Backprop: the inference method that fits a mean-field Gaussian posterior over part of the network's weights by minimizing the negative ELBO.
_Avoid_: BNN (names the model family, not the method)

**HNN**:
The heteroscedastic network with no posterior stage: mean and aleatoric variance only, the baseline without epistemic uncertainty.
_Avoid_: Deterministic baseline

## Comparison

**Comparison protocol**:
The fixed sequence of stages, data, seeds and metrics under which inference methods are compared, such that they differ only in the posterior stage.
_Avoid_: Benchmark, experiment setup

**Backbone run**:
The run that trains the mean and variance stages once per seed of the comparison protocol, shared by every method run of that seed.
_Avoid_: Base run, parent run (a parent is an MLflow relation, not a role)

**Method run**:
The run that restores a backbone run and adds one inference method's posterior stage (none for HNN), and that holds that method's evaluation.
_Avoid_: Child run, posterior run, evaluation run
