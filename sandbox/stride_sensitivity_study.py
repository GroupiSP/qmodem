"""PROTOTYPE (throwaway): stride sensitivity of the posterior stage.

Question (``sandbox/stride_sensitivity_study.md``): what does reducing the overlap of
the training windows (stride 1 = 90% overlap at W = 10) to 50%, 20% and 10% do to
the posterior stage?

Per overlap, one MLflow experiment ``<prefix>-overlap-<pct>`` holding, per seed:

- a backbone run (mean + variance) trained on the strided training windows,
- an HNN method run and a BBB method run on that backbone (N = strided windows),
- per BBB epoch (``probe/...`` metrics): the epistemic/aleatoric ratio
  ``Var_m(mu_m) / s**2`` on the posterior validation set, and the CRPS of one test
  case relative to the simulator floor ``1/2 E|Y - Y'|``,
- on both method runs, the test-case evaluation (CRPS, RMSE, coverage@0.95), the
  floor-relative CRPS, and on BBB the ratios to HNN.

Assumptions (state them when reading the results):

- overlap = (W - stride) / W, so stride = round(W * (1 - overlap)): 50% -> 5,
  20% -> 8, 10% -> 9. Only the TRAINING windows are strided; both validation sets
  keep their K fixed validation timestamps, so validation scores stay comparable.
- The backbone is re-trained on the strided windows too (HNN = same backbone).
- HPs are frozen from the newest finished protocol HPO studies (tuned at stride 1).
- The test case is the shorter-RUL policy's (policy 1) case of median length.
- "CRPS noise" is [to refine]: std of the residual around a rolling median of the
  per-epoch curve, also relative to the curve's median.

Example:
    uv run sandbox/stride_sensitivity_study.py
    uv run sandbox/stride_sensitivity_study.py --overlaps 0.9 0.5 0.2 0.1 --seeds 0 1
    # smoke
    uv run sandbox/stride_sensitivity_study.py --prefix PROTOTYPE-smoke-stride \
        --mean-epochs 2 --variance-epochs 2 --posterior-epochs 4 --patience 2 \
        --n-reference-samples 20
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Any

import jax
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
from mlflow.tracking import MlflowClient
from stagereg.core import TrainingEvent

from qmodem.battery.data_generation import (
    load_simulation_config,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)
from qmodem.battery.data_processing import (
    FEATURES,
    WINDOW_SIZE,
    PreparedData,
    SplitSizes,
    make_data_pipeline,
    prepare_data,
    split_histories,
)
from qmodem.battery.evaluate import (
    EvaluatedCase,
    EvaluationParameters,
    case_metrics,
    evaluate_predictive_model,
    log_evaluation,
    predict_case,
)
from qmodem.battery.hpo import frozen_backbone_hyperparameters
from qmodem.battery.method_run import (
    MethodConfig,
    MethodRun,
    load_backbone_handle,
    method_config,
    method_key,
    method_tags,
    posterior_overrides,
    train_method_run,
)
from qmodem.battery.posterior import (
    PREDICTIVE_SAMPLE_COUNT,
    PosteriorKeys,
    PosteriorSpec,
)
from qmodem.battery.posterior_hpo import frozen_method_config
from qmodem.battery.train import (
    MeanHyperparameters,
    Method,
    StageScheduleHyperparameters,
    VarianceHyperparameters,
    track_backbone_run,
)
from qmodem.data import (
    ArrayDataSource,
    DataPipeline,
    ScalingMode,
    ScalingStep,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    to_jax,
)
from qmodem.script_session import script_session
from qmodem.tracking import (
    MLflowTracker,
    ScalerArtifact,
    Stage,
    log_window_counts,
    mlflow_checkpoint_store,
    track_mlflow,
)

logger = logging.getLogger(__name__)

EVENT_TIME = 900.0  # as scripts/battery_multiple_scenarios/evaluate.py
SHORT_RUL_POLICY = 1  # policy 1: mean test length ~593 rows vs ~780 for policy 0
PROBE = "probe"
VAL_CRPS = f"{Stage.POSTERIOR}/validation/crps"


# --- data ------------------------------------------------------------------------


def stride_of(overlap: float, window_size: int = WINDOW_SIZE) -> int:
    """Return the stride giving ``overlap`` between consecutive windows.

    Args:
        overlap: Fraction of a window shared with the next one, in ``[0, 1)``.
        window_size: Window length W.

    Returns:
        ``round(W * (1 - overlap))``, at least 1.
    """
    return max(1, round(window_size * (1.0 - overlap)))


def strided_train(
    raw_dir: pathlib.Path, data_gen_run_id: str, stride: int
) -> tuple[PreparedData, skpp.StandardScaler, skpp.MinMaxScaler]:
    """Prepare the protocol data, then re-window the training histories at ``stride``.

    Scalers act on rows before windowing, so they are the same at any stride.

    Args:
        raw_dir: Directory holding ``train.csv``.
        data_gen_run_id: The data-generation run.
        stride: Stride of the training windows.

    Returns:
        The prepared data (strided train, protocol validation sets) and scalers.
    """
    x_scaler, y_scaler = skpp.StandardScaler(), skpp.MinMaxScaler(feature_range=(0, 1))
    data = prepare_data(
        raw_dir,
        data_gen_run_id,
        make_data_pipeline(WINDOW_SIZE, x_scaler=x_scaler, y_scaler=y_scaler),
    )
    splits = split_histories(
        pd.read_csv(raw_dir / "train.csv"),
        SplitSizes.from_params(mlflow.get_run(data_gen_run_id).data.params),
    )
    pipeline = DataPipeline(
        [
            ScalingStep(scaler=x_scaler, features=list(FEATURES)),
            ScalingStep(scaler=y_scaler, features=["rul"]),
            functools.partial(
                get_time_windows_and_join,
                window_size=WINDOW_SIZE,
                stride=stride,
                features=list(FEATURES),
            ),
            add_feature_dimension_to_y,
            to_jax,
        ]
    )
    pipeline.set_mode(ScalingMode.TRANSFORM)
    X, y = pipeline(splits.train)
    train = ArrayDataSource(features=X, targets=y)
    return dataclasses.replace(data, train=train), x_scaler, y_scaler


# --- test case and CRPS floor ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TestCase:
    """The single evaluated test case and its simulator reference.

    Attributes:
        case_id: Its ``run_id`` in ``test.csv``.
        history: Its rows.
        reference: Reference RUL samples ``(T, R)``, t0 included.
        floor: Per-timestamp CRPS floor ``1/2 E|Y - Y'|`` ``(T - 1,)``, t0 dropped.
    """

    case_id: int
    history: pd.DataFrame
    reference: np.ndarray
    floor: np.ndarray

    @property
    def key(self) -> str:
        return f"test_case_{self.case_id}"


def crps_floor(samples: np.ndarray) -> np.ndarray:
    """Return ``1/2 E|Y - Y'|`` per row, unbiased over distinct sample pairs.

    It is the expected CRPS of the true distribution against itself, the lowest
    expected CRPS any forecast can reach.

    Args:
        samples: Samples ``(T, R)``, ``R >= 2``.

    Returns:
        The floor of each row ``(T,)``.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    r = s.shape[1]
    weights = 2 * np.arange(1, r + 1) - r - 1  # sum_{i<j} (s_j - s_i)
    return (weights * s).sum(axis=1) / (r * (r - 1))


def median_length_case(test: pd.DataFrame, policy: int) -> int:
    """Return the ``run_id`` of the median-length test case of ``policy``.

    Args:
        test: The test histories.
        policy: The current policy.

    Returns:
        The case's ``run_id`` (lower median for an even count).
    """
    lengths = test[test["policy_id"] == policy].groupby("run_id").size().sort_values()
    return int(lengths.index[(len(lengths) - 1) // 2])


def build_test_case(
    test: pd.DataFrame,
    case_id: int,
    data_gen_run_id: str,
    parameters: EvaluationParameters,
    n_reference_samples: int,
) -> TestCase:
    """Simulate one test case's reference samples and CRPS floor.

    Args:
        test: The test histories.
        case_id: The case's ``run_id``.
        data_gen_run_id: The data-generation run (simulator config).
        parameters: Evaluation parameters (timestamps).
        n_reference_samples: Simulator samples per timestamp.

    Returns:
        The test case.
    """
    history = test[test["run_id"] == case_id]
    reference = reconstruct_true_rul_distribution(
        test_data=history,
        simulator_base_config=load_simulation_config(run_id=data_gen_run_id),
        n_soc0s=parameters.n_timestamps,
        n_mc_samples=n_reference_samples,
        simulator_updater=sim_updater_two_scenarios,
        event_fn=lambda t0, soc0: t0 > EVENT_TIME,
    )[f"test_case_{case_id}"]
    reference = np.asarray(reference, dtype=np.float64)
    return TestCase(case_id, history, reference, crps_floor(reference[1:]))


def floor_relative(case: EvaluatedCase, test_case: TestCase) -> dict[str, float]:
    """Return the case's CRPS relative to the floor (ratio of means, and mean ratio).

    Args:
        case: The evaluated case.
        test_case: Its floor.

    Returns:
        ``crps_over_floor`` and ``crps_over_floor_mean_ratio``.
    """
    crps = case.predictive.crps(case.reference)
    floor = np.maximum(test_case.floor, 1e-12)
    return {
        "crps_floor": float(test_case.floor.mean()),
        "crps_over_floor": float(crps.mean() / floor.mean()),
        "crps_over_floor_mean_ratio": float(np.mean(crps / floor)),
    }


# --- per-epoch probe -----------------------------------------------------------


@dataclasses.dataclass
class EpochProbe:
    """Event sink logging the epistemic/aleatoric ratio and test CRPS every epoch.

    ``method`` is set when the posterior stage builds its inference method
    (``ProbedConfig``); ``hnn`` supplies the frozen backbone networks. Metrics go
    to the active MLflow run (the BBB method run).
    """

    client: MlflowClient
    hnn: MethodRun
    val_inputs: jax.Array
    val_key: jax.Array
    test_case: TestCase
    x_scaler: Any
    y_scaler: Any
    parameters: EvaluationParameters
    method: Any = None
    history: list[dict[str, float]] = dataclasses.field(default_factory=list)

    def on_event(self, event: TrainingEvent) -> None:
        """Probe the current posterior at each posterior ``epoch_end``.

        Args:
            event: A training event; others are ignored.
        """
        if event.name != "epoch_end" or event.stage != Stage.POSTERIOR:
            return
        if self.method is None or not self.method.has_posterior:
            return
        run = dataclasses.replace(
            self.hnn, method=Method.BBB, posterior=self.method.posterior()
        )
        predictive = run.predictive(
            self.val_inputs, self.val_key, PREDICTIVE_SAMPLE_COUNT
        )
        draws = np.asarray(predictive.draws, dtype=np.float64)[..., 0]  # (M, B)
        epistemic = draws.var(axis=0)
        aleatoric = np.broadcast_to(
            np.asarray(predictive.aleatoric_variance, dtype=np.float64)[:, 0],
            epistemic.shape,
        )
        case = predict_case(
            run,
            self.test_case.history,
            self.test_case.reference,
            name=f"case_{self.test_case.case_id}",
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            parameters=self.parameters,
            key=self.val_key,
        )
        metrics = {
            "val/epistemic_var": float(epistemic.mean()),
            "val/aleatoric_var": float(aleatoric.mean()),
            "val/epi_over_ale": float(epistemic.mean() / aleatoric.mean()),
            "val/epi_over_ale_median": float(np.median(epistemic / aleatoric)),
            "test_case/crps": float(case.predictive.crps(case.reference).mean()),
            **{
                f"test_case/{k}": v
                for k, v in floor_relative(case, self.test_case).items()
            },
        }
        self.history.append({"epoch": event.step, **metrics})
        MLflowTracker(mlflow.active_run().info.run_id, client=self.client).log_metrics(
            {f"{PROBE}/{k}": v for k, v in metrics.items()}, step=event.step
        )


@dataclasses.dataclass(frozen=True)
class ProbedConfig:
    """A method configuration whose inference method is handed to an ``EpochProbe``."""

    inner: MethodConfig
    probe: EpochProbe

    @property
    def method(self) -> Method:
        return self.inner.method

    def posterior_hyperparameters(self) -> dict[str, Any]:
        return self.inner.posterior_hyperparameters()

    def posterior_spec(self) -> PosteriorSpec | None:
        spec = self.inner.posterior_spec()
        if spec is None:
            return None

        def build(key: jax.Array) -> Any:
            self.probe.method = spec.build_method(key)
            return self.probe.method

        return dataclasses.replace(spec, build_method=build)


# --- runs ------------------------------------------------------------------------


def latest_study(client: MlflowClient, experiment_id: str, filter_: str) -> str:
    """Return the newest finished HPO study run matching ``filter_``.

    Args:
        client: The MLflow client.
        experiment_id: The protocol experiment.
        filter_: Extra MLflow filter.

    Returns:
        The study run ID.

    Raises:
        SystemExit: If none matches.
    """
    runs = client.search_runs(
        [experiment_id],
        f"tags.role = 'hpo_study' AND attributes.status = 'FINISHED' AND {filter_}",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No finished HPO study with {filter_!r}.")
    return runs[0].info.run_id


def run_method(
    *,
    backbone_run_id: str,
    data: PreparedData,
    config: MethodConfig,
    seed: int,
    setup: Any,
    client: MlflowClient,
    study_run_id: str | None,
    event_sinks: Sequence[Any] = (),
) -> tuple[str, MethodRun]:
    """``track_method_run`` with extra event sinks (it has no such argument).

    Returns:
        The run ID and the trained method run.
    """
    tags = method_tags(
        method=config.method,
        seed=seed,
        backbone_run_id=backbone_run_id,
        study_run_id=study_run_id,
    )
    with track_mlflow(
        dataclasses.replace(
            setup,
            run_name=f"{config.method.value}-seed-{seed}",
            tags={**setup.tags, **tags},
        )
    ) as run:
        run_id = run.info.run_id
        method_run = train_method_run(
            backbone=load_backbone_handle(backbone_run_id, client=client),
            data=data,
            config=config,
            seed=seed,
            tracker=MLflowTracker(run_id, client=client),
            store=mlflow_checkpoint_store(run_id, client=client),
            event_sinks=event_sinks,
        )
    return run_id, method_run


def evaluate_one(
    run_id: str,
    run: MethodRun,
    test_case: TestCase,
    x_scaler: Any,
    y_scaler: Any,
    parameters: EvaluationParameters,
    client: MlflowClient,
) -> dict[str, float]:
    """Evaluate a method run on the test case and log it into the run.

    Returns:
        CRPS, RMSE, coverages, WSU and the floor-relative CRPS.
    """
    evaluation = evaluate_predictive_model(
        run,
        test_histories=test_case.history,
        reference_samples={test_case.key: test_case.reference},
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameters=parameters,
        key=run.evaluation_key,
    )
    tracker = MLflowTracker(run_id, client=client)
    log_evaluation(evaluation, tracker)
    case = evaluation.cases[0]
    metrics = case_metrics(case, parameters.coverage_levels, parameters.wsu_level)
    metrics |= floor_relative(case, test_case)
    tracker.log_metrics({f"{PROBE}/eval/{k}": v for k, v in metrics.items()}, step=0)
    return metrics


def rolling_noise(values: np.ndarray, window: int = 9) -> tuple[float, float]:
    """[to refine] Std of the residual around a centered rolling median.

    Args:
        values: A per-epoch curve.
        window: Rolling-median window.

    Returns:
        The residual std and that std over the curve's median (NaN if too short).
    """
    if len(values) < window:
        return float("nan"), float("nan")
    trend = pd.Series(values).rolling(window, center=True).median()
    residual = (pd.Series(values) - trend).dropna()
    noise = float(residual.std())
    return noise, noise / float(np.median(values))


def run_seed(
    *,
    args: argparse.Namespace,
    overlap: float,
    seed: int,
    session: Any,
    studies: dict[str, str],
    test_case: TestCase,
    parameters: EvaluationParameters,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Train backbone, HNN and BBB for one overlap and seed, and evaluate them.

    Returns:
        The summary row and the BBB per-epoch probe history (with val CRPS).
    """
    client = session.client
    stride = stride_of(overlap)
    data_gen_run_id = os.environ["DATA_GEN_RUN_ID_MULTI"]
    data, x_scaler, y_scaler = strided_train(
        pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]), data_gen_run_id, stride
    )
    logger.info(
        "overlap %.0f%% stride %d: %s", 100 * overlap, stride, data.window_counts()
    )
    setup = dataclasses.replace(
        session.setup,
        experiment_name=f"{args.prefix}-overlap-{round(100 * overlap)}",
        tags={
            "prototype": "stride_sensitivity",
            "overlap": str(overlap),
            "stride": str(stride),
            "test_case": str(test_case.case_id),
        },
    )

    mean_hp, variance_hp = frozen_backbone_hyperparameters(
        mean_study_run_id=studies["mean"],
        variance_study_run_id=studies["variance"],
        mean_base=MeanHyperparameters(
            schedule=StageScheduleHyperparameters(max_epochs=args.mean_epochs)
        ),
        variance_base=VarianceHyperparameters(
            schedule=StageScheduleHyperparameters(max_epochs=args.variance_epochs)
        ),
    )
    backbone = track_backbone_run(
        data=data,
        mean_hp=mean_hp,
        variance_hp=variance_hp,
        seed=seed,
        data_generation_run_id=data_gen_run_id,
        scalers=(
            ScalerArtifact("x_scaler", x_scaler, np.ones((WINDOW_SIZE, len(FEATURES)))),
            ScalerArtifact("y_scaler", y_scaler, np.ones((1, 1)), "inverse_transform"),
        ),
        setup=dataclasses.replace(setup, run_name=f"backbone-seed-{seed}"),
        mean_study_run_id=studies["mean"],
        variance_study_run_id=studies["variance"],
    )
    log_window_counts(
        backbone.run_id, data.window_counts(), logger=logger, client=client
    )

    common = {
        "backbone_run_id": backbone.run_id,
        "data": data,
        "seed": seed,
        "setup": setup,
        "client": client,
    }
    hnn_id, hnn = run_method(
        config=method_config(Method.HNN, {}), study_run_id=None, **common
    )

    overrides = posterior_overrides(
        [], max_epochs=args.posterior_epochs, patience=args.patience
    )
    bbb_config = frozen_method_config(
        studies["bbb"], method_config(Method.BBB, overrides), client=client
    )
    val_key = PosteriorKeys.of(method_key(seed, Method.BBB)).validation
    probe = EpochProbe(
        client=client,
        hnn=hnn,
        val_inputs=data.posterior_val.features,
        val_key=val_key,
        test_case=test_case,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameters=parameters,
    )
    bbb_id, bbb = run_method(
        config=ProbedConfig(bbb_config, probe),
        study_run_id=studies["bbb"],
        event_sinks=(probe,),
        **common,
    )

    hnn_eval = evaluate_one(
        hnn_id, hnn, test_case, x_scaler, y_scaler, parameters, client
    )
    bbb_eval = evaluate_one(
        bbb_id, bbb, test_case, x_scaler, y_scaler, parameters, client
    )
    relative = {
        "crps_rel_hnn": bbb_eval["crps"] / hnn_eval["crps"],
        "rmse_rel_hnn": bbb_eval["rmse"] / hnn_eval["rmse"],
        "coverage_95_minus_hnn": bbb_eval["coverage_95"] - hnn_eval["coverage_95"],
    }

    val_crps = np.array([m.value for m in client.get_metric_history(bbb_id, VAL_CRPS)])
    test_rel = np.array([h["test_case/crps_over_floor"] for h in probe.history])
    val_noise, val_noise_rel = rolling_noise(val_crps)
    test_noise, test_noise_rel = rolling_noise(test_rel)
    noise = {
        "noise/val_crps": val_noise,
        "noise/val_crps_rel": val_noise_rel,
        "noise/test_crps_over_floor": test_noise,
        "noise/test_crps_over_floor_rel": test_noise_rel,
    }
    MLflowTracker(bbb_id, client=client).log_metrics(
        {f"{PROBE}/{k}": v for k, v in {**relative, **noise}.items()}, step=0
    )

    best = bbb.checkpoint.epoch if bbb.checkpoint is not None else -1
    at_best = next((h for h in probe.history if h["epoch"] == best), {})
    row = {
        "overlap": overlap,
        "stride": stride,
        "seed": seed,
        "N_train": len(data.train),
        "bbb_epochs": len(val_crps),
        "bbb_best_epoch": best,
        "val_crps_best": float(val_crps.min()) if len(val_crps) else float("nan"),
        "epi_over_ale@best": at_best.get("val/epi_over_ale", float("nan")),
        "crps_floor": hnn_eval["crps_floor"],
        "hnn_crps/floor": hnn_eval["crps_over_floor"],
        "bbb_crps/floor": bbb_eval["crps_over_floor"],
        "hnn_rmse": hnn_eval["rmse"],
        "bbb_rmse": bbb_eval["rmse"],
        "hnn_cov95": hnn_eval["coverage_95"],
        "bbb_cov95": bbb_eval["coverage_95"],
        **relative,
        **noise,
        "backbone_run": backbone.run_id,
        "hnn_run": hnn_id,
        "bbb_run": bbb_id,
    }
    curves = [
        {**h, "val_crps": float(val_crps[i]) if i < len(val_crps) else float("nan")}
        for i, h in enumerate(probe.history)
    ]
    return row, curves


def plot_curves(curves: dict[float, list[dict[str, float]]], out: pathlib.Path) -> None:
    """Plot val CRPS, epistemic/aleatoric ratio and test CRPS/floor per overlap.

    Args:
        curves: Per overlap, the BBB per-epoch history (first seed).
        out: Output PNG path.
    """
    panels = [
        ("val_crps", "posterior val CRPS (scaled)"),
        ("val/epi_over_ale", "Var_m(mu_m) / s^2 (val)"),
        ("test_case/crps_over_floor", "test-case CRPS / floor"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4))
    for overlap, history in sorted(curves.items()):
        epochs = [h["epoch"] for h in history]
        for ax, (key, title) in zip(axes, panels):
            ax.plot(epochs, [h[key] for h in history], label=f"{100 * overlap:.0f}%")
            ax.set_title(title)
            ax.set_xlabel("BBB epoch")
    axes[1].set_yscale("log")
    axes[0].legend(title="overlap")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Command-line arguments.

    Returns:
        The parsed arguments.
    """
    defaults = StageScheduleHyperparameters()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.5, 0.2, 0.1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--prefix", default="stride-sensitivity")
    parser.add_argument("--mean-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--variance-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--posterior-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--case-id", type=int, default=None)
    parser.add_argument("--n-timestamps", type=int, default=20)
    parser.add_argument("--num-draws", type=int, default=100)
    parser.add_argument("--n-reference-samples", type=int, default=100)
    parser.add_argument("--mean-study-run-id", default=None)
    parser.add_argument("--variance-study-run-id", default=None)
    parser.add_argument("--bbb-study-run-id", default=None)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("sandbox/stride_sensitivity_out"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the stride sensitivity study.

    Args:
        argv: Command-line arguments.
    """
    args = parse_args(argv)
    parameters = EvaluationParameters(
        n_timestamps=args.n_timestamps, num_draws=args.num_draws
    )
    with script_session() as session:
        protocol_exp = session.require_experiment_ids()[0]
        studies = {
            "mean": args.mean_study_run_id
            or latest_study(session.client, protocol_exp, "tags.stage = 'mean'"),
            "variance": args.variance_study_run_id
            or latest_study(session.client, protocol_exp, "tags.stage = 'variance'"),
            "bbb": args.bbb_study_run_id
            or latest_study(
                session.client,
                protocol_exp,
                "tags.stage = 'posterior' AND tags.method = 'bbb'",
            ),
        }
        logger.info("Frozen HPs from studies %s", studies)

        test = pd.read_csv(pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]) / "test.csv")
        case_id = (
            args.case_id
            if args.case_id is not None
            else median_length_case(test, SHORT_RUL_POLICY)
        )
        test_case = build_test_case(
            test,
            case_id,
            os.environ["DATA_GEN_RUN_ID_MULTI"],
            parameters,
            args.n_reference_samples,
        )
        logger.info(
            "Test case %d (policy %d): CRPS floor per timestamp %s",
            case_id,
            SHORT_RUL_POLICY,
            np.round(test_case.floor, 1),
        )

        rows, curves = [], {}
        for overlap in args.overlaps:
            for seed in args.seeds:
                row, history = run_seed(
                    args=args,
                    overlap=overlap,
                    seed=seed,
                    session=session,
                    studies=studies,
                    test_case=test_case,
                    parameters=parameters,
                )
                rows.append(row)
                curves.setdefault(overlap, history)
                logger.info("Row: %s", row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "summary.csv", index=False)
    plot_curves(curves, args.out_dir / "curves.png")
    shown = table.drop(columns=["backbone_run", "hnn_run", "bbb_run"])
    print(shown.T.to_string(float_format="{:.4g}".format))
    print(f"\nWrote {args.out_dir / 'summary.csv'} and {args.out_dir / 'curves.png'}")


if __name__ == "__main__":
    main()
