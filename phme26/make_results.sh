#!/usr/bin/env bash
# ==========================================================================
# PHMe26 — Reproduce conference results.
# Running this script requires uv (https://docs.astral.sh/uv/).
# To install uv, run `pip install uv` or `pipx install uv`.
#
# Usage:
#   cd qmodem/
#   bash phme26/make_results.sh           # skip data generation
#   bash phme26/make_results.sh --gen     # regenerate data first
#
# Prerequisites:
#   uv sync                   # install dependencies (once)
# ==========================================================================
set -euo pipefail

# Run every `qmodem` invocation through uv so the project venv is used.
qmodem() { uv run qmodem "$@"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"

DATA_DIR="$BASE_DIR/data"
TRAINED_DIR="$BASE_DIR/trained"
RESULTS_DIR="$BASE_DIR/results"

N_TEST_CASES=4
METHODS="het_cnn mcd_cnn bayes_cnn qavi_cnn"

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
GENERATE=false
for arg in "$@"; do
    case "$arg" in
        --gen) GENERATE=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 1. Data generation (optional)
# ------------------------------------------------------------------
if [ "$GENERATE" = true ]; then
    echo "============================================================"
    echo "Generating training, validation, and test data …"
    echo "============================================================"
    qmodem generate-data \
        --n-test-cases "$N_TEST_CASES" \
        --output-dir "$DATA_DIR"
    echo
fi

# Verify data exists
if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo "ERROR: $DATA_DIR/train.npz not found."
    echo "Run with --gen to generate data first."
    exit 1
fi

# ------------------------------------------------------------------
# 2. Train all methods
# ------------------------------------------------------------------
for method in $METHODS; do
    echo "============================================================"
    echo "Training: $method"
    echo "============================================================"
    qmodem train "$method" \
        --train-data-path "$DATA_DIR/train.npz" \
        --val-data-path "$DATA_DIR/val.npz" \
        --output-dir "$TRAINED_DIR"
    echo
done

# ------------------------------------------------------------------
# 3. Compare all methods on each test case
# ------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"

for i in $(seq 0 $((N_TEST_CASES - 1))); do
    echo "============================================================"
    echo "Comparing all methods on test_case_${i}"
    echo "============================================================"
    qmodem compare \
        --test-data-path "$DATA_DIR/test_case_${i}.npz" \
        --trained-dir "$TRAINED_DIR" \
        --output-dir "$RESULTS_DIR"
    echo
done

echo "============================================================"
echo "All done! Results saved in $RESULTS_DIR"
echo "============================================================"
