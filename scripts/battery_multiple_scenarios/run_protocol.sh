#!/usr/bin/env bash
# Run the comparison protocol end to end (docs/comparison-protocol.md).
#
# Steps: mean study -> provisional seed-0 backbone -> variance study -> tuned seed-0
# backbone -> one posterior study per method -> per seed: backbone, method runs,
# evaluation. Each step's run ID is read back from MLflow (newest finished run
# with the step's tags, started after the step began) and passed explicitly to the
# next step, so tag selection never picks a stale run.
#
# The data-generation run must exist already (DATA_GEN_RUN_ID_MULTI in .env).
#
# Environment knobs:
#   SEEDS    protocol seeds (default "0 1 2 3 4")
#   METHODS  methods with a posterior stage (default "mcd bbb psgld q-avi";
#            add c-avi for the ablation arm)
#   SMOKE=1  tiny budgets and epoch caps, to check the pipeline end to end
#            (runs are tagged protocol_overrides=true)
#
# Example:
#   bash scripts/battery_multiple_scenarios/run_protocol.sh
#   SMOKE=1 SEEDS="0 1" METHODS="mcd psgld" \
#       bash scripts/battery_multiple_scenarios/run_protocol.sh

set -euo pipefail

if ((BASH_VERSINFO[0] < 4)); then
    echo "Needs bash >= 4 (associative arrays); macOS: brew install bash." >&2
    exit 1
fi

S=scripts/battery_multiple_scenarios
SEEDS=${SEEDS:-"0 1 2 3 4"}
METHODS=${METHODS:-"mcd bbb psgld q-avi"}

if [[ ${SMOKE:-0} == 1 ]]; then
    MEAN_STUDY_ARGS=(--n-trials 2 --max-epochs 2)
    BACKBONE_ARGS=(--mean-epochs 2 --variance-epochs 2)
    VARIANCE_STUDY_ARGS=(--n-trials 2 --max-epochs 2)
    POSTERIOR_ARGS=(--max-epochs 4 --patience 2)
    POSTERIOR_STUDY_ARGS=(--n-trials 2)
    PSGLD_ARGS=(--posterior-param burn_in_epochs=2)
    EVALUATE_ARGS=(--n-timestamps 3 --num-samples 10 --n-reference-samples 10)
else
    MEAN_STUDY_ARGS=()
    BACKBONE_ARGS=()
    VARIANCE_STUDY_ARGS=()
    POSTERIOR_ARGS=()
    POSTERIOR_STUDY_ARGS=()
    PSGLD_ARGS=()
    EVALUATE_ARGS=()
fi

log() { printf '\n=== %s\n' "$*" >&2; }

now_ms() { uv run python -c 'import time; print(int(time.time() * 1000))'; }

# newest_run SINCE_MS FILTER: ID of the newest finished run matching FILTER
# (an MLflow filter string) that started at or after SINCE_MS.
newest_run() {
    uv run python - "$1" "$2" <<'PY'
import sys

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from qmodem.tracking import MLFlowSetup

load_dotenv(".env")  # find_dotenv() fails on a script read from stdin
since, filter_string = int(sys.argv[1]), sys.argv[2]
setup = MLFlowSetup()
client = MlflowClient(tracking_uri=setup.backend_store)
experiment = client.get_experiment_by_name(setup.experiment_name)
if experiment is None:
    sys.exit(f"No experiment {setup.experiment_name!r}.")
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string=(
        f"{filter_string} AND attributes.status = 'FINISHED' "
        f"AND attributes.start_time >= {since}"
    ),
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    sys.exit(f"No finished run matches {filter_string!r} since {since}.")
print(runs[0].info.run_id)
PY
}

# run_step FILTER CMD...: run CMD, then print the ID of the run it created.
run_step() {
    local filter=$1 since
    shift
    since=$(now_ms)
    "$@" >&2
    newest_run "$since" "$filter"
}

method_args() {
    if [[ $1 == psgld ]]; then
        echo "${POSTERIOR_ARGS[@]}" "${PSGLD_ARGS[@]}"
    else
        echo "${POSTERIOR_ARGS[@]}"
    fi
}

log "Mean study"
MEAN_ID=$(run_step "tags.role = 'hpo_study' AND tags.stage = 'mean'" \
    uv run $S/run_mean_study.py "${MEAN_STUDY_ARGS[@]}")
log "mean study: $MEAN_ID"

log "Provisional seed-0 backbone"
BB0_PROV=$(run_step "tags.role = 'backbone' AND tags.seed = '0'" \
    uv run $S/train_backbone.py --seed 0 --mean-study-run-id "$MEAN_ID" \
    "${BACKBONE_ARGS[@]}")
log "provisional backbone: $BB0_PROV"

log "Variance study"
VAR_ID=$(run_step "tags.role = 'hpo_study' AND tags.stage = 'variance'" \
    uv run $S/run_variance_study.py --backbone-run-id "$BB0_PROV" \
    "${VARIANCE_STUDY_ARGS[@]}")
log "variance study: $VAR_ID"

train_backbone() {
    run_step "tags.role = 'backbone' AND tags.seed = '$1'" \
        uv run $S/train_backbone.py --seed "$1" \
        --mean-study-run-id "$MEAN_ID" --variance-study-run-id "$VAR_ID" \
        "${BACKBONE_ARGS[@]}"
}

log "Tuned seed-0 backbone"
BB0=$(train_backbone 0)
log "seed-0 backbone: $BB0"

declare -A STUDY_ID
for m in $METHODS; do
    log "Posterior study: $m"
    # shellcheck disable=SC2046  # method_args is word-split on purpose
    STUDY_ID[$m]=$(run_step \
        "tags.role = 'hpo_study' AND tags.stage = 'posterior' AND tags.method = '$m'" \
        uv run $S/run_posterior_study.py --method "$m" --backbone-run-id "$BB0" \
        "${POSTERIOR_STUDY_ARGS[@]}" $(method_args "$m"))
    log "$m study: ${STUDY_ID[$m]}"
done

for r in $SEEDS; do
    if [[ $r == 0 ]]; then
        BB=$BB0
    else
        log "Backbone seed $r"
        BB=$(train_backbone "$r")
    fi
    log "seed $r backbone: $BB"

    for m in hnn $METHODS; do
        log "Method run: $m, seed $r"
        if [[ $m == hnn ]]; then
            extra=()
        else
            # shellcheck disable=SC2207
            extra=(--study-run-id "${STUDY_ID[$m]}" $(method_args "$m"))
        fi
        RUN_ID=$(run_step \
            "tags.role = 'method' AND tags.method = '$m' AND tags.seed = '$r'" \
            uv run $S/train_method_run.py --method "$m" --backbone-run-id "$BB" \
            "${extra[@]}")
        log "Evaluate: $m, seed $r ($RUN_ID)"
        uv run $S/evaluate.py --run-id "$RUN_ID" "${EVALUATE_ARGS[@]}" >&2
    done
done

log "Protocol done: mean study $MEAN_ID, variance study $VAR_ID"
for m in $METHODS; do log "$m study: ${STUDY_ID[$m]}"; done
