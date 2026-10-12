# Can the battery multiple-scenarios dataloaders satisfy stagereg's protocols?

Research for GroupiSP/qmodem#91 (map: #89). Sources: qmodem at `fe4a733`, stagereg local clone (`src/stagereg/core/protocols.py`, `core/types.py`, `jax/supervised_staged.py`, `jax/posterior_stage.py`, `jax/validation.py`, `jax/evaluation.py`, `jax/epoch_loop.py`, `examples/jax/cmapss/run.py`).

## Verdict

**Not as-is, but only a thin adapter is needed (~20 lines).** The data path up to `PreparedData` (mlflow run ID -> `train.csv` split by `run_id` -> `DataPipeline` with train-fitted scalers -> windowed `jax.Array`s) can be reused unchanged. What has to change is the last step. `dataloader_builders` returns two `Callable[[int], Iterable[tuple]]` builders, and stagereg wants one `LoaderFactory(*, split, epoch) -> Iterable[Batch]`. The shuffling semantics already match: the order is deterministic per epoch and derived from the epoch index. The real incompatibilities are in the models, not the data (see "Violated assumptions").

## Protocol-by-protocol fit

| stagereg contract | qmodem today | Fit | Gap |
|---|---|---|---|
| `Dataset` (`__len__`, `get(i) -> Mapping`) (`core/protocols.py:17`) | `ArrayDataSource` has `__len__` and `__getitem__ -> (x, y)` (`data.py`) | No (no `get`, returns a tuple) | **Irrelevant.** Nothing in `stagereg.jax` consumes `Dataset`; it is only exported (`core/__init__.py:60`). Skip it. |
| `LoaderFactory.__call__(*, split, epoch) -> Iterable[Batch]` (`core/protocols.py:26`) | `train_builder(epoch)` / `val_builder(epoch)`, positional `int`, separate per split (`battery/data_processing.py:59`) | No | Need keyword-only `split`/`epoch` and dispatch on `split` in one callable. |
| `Batch(inputs, targets, sample_weight, metadata)` frozen dataclass (`core/types.py:45`) | grain yields `tuple[np.ndarray, np.ndarray]`; val yields `[(X_val, y_val)]` (verified at runtime) | No | Wrap each tuple in `Batch(inputs=x, targets=y)`. Shapes already fit: `x: (B, W, 2)`, `y: (B, 1)`. |
| Re-iterability / per-epoch determinism (cmapss: `permutation(key(epoch))`) | `IndexSampler(shuffle=True, seed=epoch, num_epochs=1)`, a fresh `grain.DataLoader` per call, so the same epoch gives the same order (verified) | Yes | None. Epochs are 0-based in both (`epoch_loop.py:220`, `qmodem/train.py:113`). |
| `HeldOutValidation(model, loader, loss)` calls `loader(split="validation", epoch=...)` (`jax/validation.py:57`) | `val_builder` returns one full-set batch and ignores `epoch` | Yes, once wrapped | Split name must be `"validation"` (qmodem uses `val`). |
| `materialize_residual_loader(mean_model, source_loader, splits=("train","validation"), source_epoch=0)` (`supervised_staged.py:48`) | n/a | Data: yes. Model: **no** | It consumes each split once and caches it. It needs non-`None` targets and `mean_model(batch.inputs).shape == batch.targets.shape` (`supervised_staged.py:107`). |
| `PosteriorStage.dataset_size: int` (N) scaling the log-likelihood by `N / B` (`posterior_stage.py:178,686`) | `len(data.train)` = number of **windows** `N_w` (already used as `n_train_samples` for BNN KL scaling, `battery/train.py`) | Yes | Pass `len(data.train)`. `B` is read from `targets.shape[0]`, so the ragged last batch (`drop_remainder=False`) is scaled correctly. |

## Adapter sketch

```python
# src/qmodem/battery/stagereg_data.py
from collections.abc import Callable, Iterable

import jax.numpy as jnp
from stagereg.core import Batch, LoaderFactory

from qmodem.battery.data_processing import PreparedData, dataloader_builders


def make_loader_factory(
    data: PreparedData, batch_size: int, drop_remainder: bool
) -> LoaderFactory:
    """Adapt qmodem's per-split epoch builders to stagereg's LoaderFactory."""
    train_b, val_b = dataloader_builders(data, batch_size, drop_remainder)
    builders: dict[str, Callable[[int], Iterable]] = {
        "train": train_b,
        "validation": val_b,
    }

    def loader(*, split: str, epoch: int) -> list[Batch]:
        if split not in builders:
            raise ValueError(f"unknown split {split!r}")
        return [
            Batch(inputs=jnp.asarray(x), targets=jnp.asarray(y))
            for x, y in builders[split](epoch)
        ]

    return loader

# usage: data = prepare_data(raw_dir, run_id, pipeline)   # unchanged
#        loader = make_loader_factory(data, hp.batch_size, hp.drop_remainder)
#        PosteriorStage(..., train_loader=loader, dataset_size=len(data.train))
```

Materializing into a `list` makes each call re-iterable and lets `materialize_residual_loader` detect an empty split. The train set is already held in memory as `jax.Array`s, so this costs nothing extra.

## Violated / notable assumptions

1. **Model call signature (blocking, not a data issue).** stagereg calls `model(batch.inputs)` everywhere: losses (`jax/losses.py:51`), residual materialization (`supervised_staged.py:106`) and the posterior. qmodem's battery `CNN.__call__(x, rngs)` requires `rngs` and returns `(B, 2)` `[mu, var]` (`battery/models.py:158`). The model must therefore be wrapped as a single-arg module returning `(B, 1)` means for `MeanStage`/residuals. Otherwise `_materialize_residual_batch` raises "mean predictions and targets must have matching shapes". stagereg's separate mean/variance models (cmapss `Cnn1DMeanModel`/`Cnn1DGammaModel`) are the intended shape.
2. **Residual snapshot freezes batch order.** `materialize_residual_loader` uses `source_epoch=0` and then replays the same cached batches every epoch (`epoch` is ignored). The variance stage therefore never reshuffles. This is acceptable by design, but it differs from qmodem's per-epoch reshuffle. It also caches the whole train and validation sets on device, which is fine at the current dataset size.
3. **Dataset size N = overlapping windows, not histories.** With `stride=1`, the windows of one discharge history are strongly correlated, so `N / B` treats `N_w` correlated windows as i.i.d. That overstates the effective sample size and makes the posterior over-confident. qmodem's current BNN KL scaling has the same issue, so switching keeps parity. Flag this for the posterior-quality ticket rather than the adapter.
4. **PRNG ownership moves to stagereg.** Loaders take no key. stagereg derives one key per batch from `state.rng_state` (`epoch_loop.py`, `evaluation.py:167`). qmodem's `train_loop` splits **one key per sample** (`batch[0].shape[0] + 1`, `qmodem/train.py:116`) and passes `subkeys` to its steps. Any ported step/loss must accept a single batch key. Shuffling stays seeded by `epoch` inside the loader, independent of `state.rng_state`.
5. **Split naming.** stagereg's defaults are `"train"`/`"validation"` (`supervised_staged.py:355`). qmodem uses `train`/`val`. The adapter maps the names.
6. **Validation is one full batch.** `evaluate_loader` averages per-batch losses, so a single batch gives an exact set mean. If validation is ever mini-batched with a ragged tail, the batch-mean of means becomes slightly biased.
7. **Stateful pipeline.** `prepare_data` mutates `ScalingStep.mode` (FIT_TRANSFORM, then TRANSFORM) and fits scalers in place. It must run exactly once before building the factory, and the scalers must be kept for inverse-transforming RUL. The adapter keeps it outside the loader, so this is fine.
8. **Ragged last batch.** With `drop_remainder=False`, the last train batch is smaller, which can trigger one extra JIT trace per stage. This is harmless.
9. **QAVI path.** `run_train_qavi.py` uses the same `prepare_data`/`dataloader_builders`, so the same adapter applies to its data. Its adversarial generator/discriminator loop has no stagereg stage counterpart, which is out of scope here.
