"""Click CLI for QMoDeM.

Usage::

    qmodem generate-data [OPTIONS]
    qmodem train METHOD [OPTIONS]
    qmodem test METHOD [OPTIONS]
    qmodem compare [OPTIONS]

Where ``METHOD`` is one of: ``bayes_cnn``, ``het_cnn``, ``mcd_cnn``,
``qavi_cnn``.
"""

from __future__ import annotations

import click

from qmodem.application import METHODS
from qmodem.utils import SHARED_PARAMS

_METHODS = click.Choice(METHODS)

_P_TRAIN = SHARED_PARAMS["training"]
_P_MODEL = SHARED_PARAMS["model"]
_P_DATA = SHARED_PARAMS["data"]
_P_GEN = SHARED_PARAMS["generate"]
_P_TEST = SHARED_PARAMS["test"]
_P_MCD = SHARED_PARAMS["mcd_cnn"]
_P_QAVI = SHARED_PARAMS["qavi_cnn"]


@click.group()
def cli() -> None:
    """QMoDeM: Quantum-aided Models for Decision Making."""


# ===================================================================
# generate-data
# ===================================================================


@cli.command("generate-data")
@click.option(
    "--n-histories-train",
    type=int,
    default=None,
    help=f"Number of training discharge histories (default: {_P_DATA['n_histories_train']}).",
)
@click.option(
    "--n-histories-val",
    type=int,
    default=None,
    help=f"Number of validation discharge histories (default: {_P_DATA['n_histories_val']}).",
)
@click.option(
    "--n-test-cases",
    type=int,
    default=_P_GEN["n_test_cases"],
    help=f"Number of independent test cases (default: {_P_GEN['n_test_cases']}).",
)
@click.option(
    "--n-simu",
    type=int,
    default=_P_GEN["n_simu"],
    help=f"Stochastic simulations per intermediate SoC for test reference RUL (default: {_P_GEN['n_simu']}).",
)
@click.option(
    "--n-intermediate-socs",
    type=int,
    default=_P_GEN["n_intermediate_socs"],
    help=f"Number of intermediate SoC evaluation points per test case (default: {_P_GEN['n_intermediate_socs']}).",
)
@click.option(
    "--seed-train", type=int, default=None, help="Random seed for training data."
)
@click.option(
    "--seed-val", type=int, default=None, help="Random seed for validation data."
)
@click.option("--seed-test", type=int, default=None, help="Random seed for test data.")
@click.option(
    "--output-dir",
    type=str,
    default="data",
    help="Output directory (default: data).",
)
def generate_data(
    n_histories_train: int | None,
    n_histories_val: int | None,
    n_test_cases: int,
    n_simu: int,
    n_intermediate_socs: int,
    seed_train: int | None,
    seed_val: int | None,
    seed_test: int | None,
    output_dir: str,
) -> None:
    """Generate train, validation, and test datasets."""
    from qmodem.application import generate_data as _generate_data

    _generate_data(
        n_histories_train=n_histories_train,
        n_histories_val=n_histories_val,
        n_test_cases=n_test_cases,
        n_simu=n_simu,
        n_intermediate_socs=n_intermediate_socs,
        seed_train=seed_train,
        seed_val=seed_val,
        seed_test=seed_test,
        output_dir=output_dir,
    )


# ===================================================================
# train
# ===================================================================


@cli.command()
@click.argument("method", type=_METHODS)
@click.option(
    "--n-epochs",
    type=int,
    default=None,
    help=f"Maximum training epochs (default: {_P_TRAIN['n_epochs']}).",
)
@click.option(
    "--lr", type=float, default=None, help=f"Learning rate (default: {_P_TRAIN['lr']})."
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help=f"Batch size (default: {_P_TRAIN['batch_size']}).",
)
@click.option(
    "--patience",
    type=int,
    default=None,
    help=f"Early stopping patience (default: {_P_TRAIN['patience']}).",
)
@click.option(
    "--print-every",
    type=int,
    default=None,
    help=f"Print progress every N epochs (default: {_P_TRAIN['print_every']}).",
)
@click.option(
    "--n-filters",
    type=int,
    default=None,
    help=f"Number of conv filters (default: {_P_MODEL['n_filters']}).",
)
@click.option(
    "--kernel-size",
    type=int,
    default=None,
    help=f"Conv kernel size (default: {_P_MODEL['kernel_size']}).",
)
@click.option(
    "--window-size",
    type=int,
    default=None,
    help=f"Time-window size (default: {_P_DATA['window_size']}).",
)
@click.option(
    "--stride",
    type=int,
    default=None,
    help=f"Stride for time windowing (default: {_P_DATA['stride']}).",
)
@click.option(
    "--no-normalize", is_flag=True, default=False, help="Disable target normalisation."
)
@click.option(
    "--train-data-path",
    type=str,
    default="data/train.npz",
    help="Path to training data.",
)
@click.option(
    "--val-data-path", type=str, default="data/val.npz", help="Path to validation data."
)
@click.option(
    "--dropout-rate",
    type=float,
    default=_P_MCD["dropout_rate"],
    help=f"Dropout rate (mcd_cnn only, default: {_P_MCD['dropout_rate']}).",
)
@click.option(
    "--n-qubits",
    type=int,
    default=_P_QAVI["n_qubits"],
    help=f"Number of qubits (qavi_cnn only, default: {_P_QAVI['n_qubits']}).",
)
@click.option(
    "--n-pqc-layers",
    type=int,
    default=_P_QAVI["n_pqc_layers"],
    help=f"Number of PQC layers (qavi_cnn only, default: {_P_QAVI['n_pqc_layers']}).",
)
def train(
    method: str,
    n_epochs: int | None,
    lr: float | None,
    batch_size: int | None,
    patience: int | None,
    print_every: int | None,
    n_filters: int | None,
    kernel_size: int | None,
    window_size: int | None,
    stride: int | None,
    no_normalize: bool,
    train_data_path: str,
    val_data_path: str,
    dropout_rate: float,
    n_qubits: int,
    n_pqc_layers: int,
) -> None:
    """Train a model using METHOD."""
    from qmodem.application import train as _train

    normalize = None if not no_normalize else False

    _train(
        method,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        print_every=print_every,
        n_filters=n_filters,
        kernel_size=kernel_size,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        dropout_rate=dropout_rate,
        n_qubits=n_qubits,
        n_pqc_layers=n_pqc_layers,
    )


# ===================================================================
# test
# ===================================================================


@cli.command()
@click.argument("method", type=_METHODS)
@click.option(
    "--test-data-path",
    type=str,
    default="data/test_case_0.npz",
    help="Path to the test-case .npz file (default: data/test_case_0.npz).",
)
@click.option(
    "--n-samples",
    type=int,
    default=_P_TEST["n_samples"],
    help=f"Number of forward passes / weight samples for uncertainty (default: {_P_TEST['n_samples']}).",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for output plots (default: saved/<method>/test).",
)
def test(
    method: str,
    test_data_path: str,
    n_samples: int,
    output_dir: str | None,
) -> None:
    """Evaluate a trained model (RUL prediction + CRPS)."""
    from qmodem.application import test as _test

    _test(
        method,
        test_data_path=test_data_path,
        n_samples=n_samples,
        output_dir=output_dir,
    )


# ===================================================================
# compare
# ===================================================================


@cli.command()
@click.option(
    "--methods",
    type=str,
    multiple=True,
    default=None,
    help=(
        "Methods to compare (repeat for each method, e.g. "
        "--methods het_cnn --methods mcd_cnn). "
        "Defaults to all methods."
    ),
)
@click.option(
    "--test-data-path",
    type=str,
    default="data/test_case_0.npz",
    help="Path to the test-case .npz file (default: data/test_case_0.npz).",
)
@click.option(
    "--n-samples",
    type=int,
    default=_P_TEST["n_samples"],
    help=f"Number of forward passes / weight samples for uncertainty (default: {_P_TEST['n_samples']}).",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for the comparison plot (default: saved/compare).",
)
def compare(
    methods: tuple[str, ...],
    test_data_path: str,
    n_samples: int,
    output_dir: str | None,
) -> None:
    """Compare multiple methods on the same test case."""
    from qmodem.application import compare as _compare

    _compare(
        methods=list(methods) if methods else None,
        test_data_path=test_data_path,
        n_samples=n_samples,
        output_dir=output_dir,
    )
