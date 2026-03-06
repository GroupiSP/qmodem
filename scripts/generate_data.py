"""Generate train and test data for battery discharge RUL experiments.
.. deprecated:: Use the ``qmodem`` CLI instead.  See ``qmodem --help``.


Usage::

    uv run python scripts/generate_data.py [OPTIONS]

All data is saved to the ``data/`` directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    SHARED_PARAMS,
    TRAIN_SEED,
    TEST_SEED,
    create_battery_and_policy,
    make_simulator_config,
)

from qmodem.generate import generate_test_data, generate_train_data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate train/val/test data for battery discharge RUL experiments."
    )
    p.add_argument(
        "--n-histories-train",
        type=int,
        default=SHARED_PARAMS["data"]["n_histories_train"],
        help="Number of training discharge histories (default: %(default)s).",
    )
    p.add_argument(
        "--n-histories-val",
        type=int,
        default=SHARED_PARAMS["data"]["n_histories_val"],
        help="Number of validation discharge histories (default: %(default)s).",
    )
    p.add_argument(
        "--n-test-cases",
        type=int,
        default=1,
        help="Number of independent test cases (default: %(default)s).",
    )
    p.add_argument(
        "--n-simu",
        type=int,
        default=500,
        help="Stochastic simulations per intermediate SoC for test reference RUL (default: %(default)s).",
    )
    p.add_argument(
        "--n-intermediate-socs",
        type=int,
        default=200,
        help="Number of intermediate SoC evaluation points per test case (default: %(default)s).",
    )
    p.add_argument(
        "--seed-train",
        type=int,
        default=TRAIN_SEED,
        help="Random seed for training data (default: %(default)s).",
    )
    p.add_argument(
        "--seed-val",
        type=int,
        default=TRAIN_SEED + 1,
        help="Random seed for validation data (default: %(default)s).",
    )
    p.add_argument(
        "--seed-test",
        type=int,
        default=TEST_SEED,
        help="Random seed for test data (default: %(default)s).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: %(default)s).",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    # Build simulator config from shared params
    sim_params = SHARED_PARAMS["simulation"]
    battery, discharge_policy = create_battery_and_policy(
        sim_params["current_amplitude"]
    )
    sim_config = make_simulator_config(
        n_simu=1,
        v_cut=sim_params["v_cut"],
        soc_0=1.0,
        dt=sim_params["dt"],
        omega_std=sim_params["omega_std"],
        eta_std=sim_params["eta_std"],
        discharge_policy=discharge_policy,
        battery=battery,
    )
    soc_range = sim_params["soc_range"]

    # --- Train data ---
    print(
        f"Generating {args.n_histories_train} training histories (seed={args.seed_train})..."
    )
    train_path = generate_train_data(
        simulator_config=sim_config,
        n_histories=args.n_histories_train,
        soc_range=soc_range,
        seed=args.seed_train,
        output_path=output_dir / "train.npz",
    )
    print(f"  Saved to {train_path}")

    # --- Validation data ---
    print(
        f"Generating {args.n_histories_val} validation histories (seed={args.seed_val})..."
    )
    val_path = generate_train_data(
        simulator_config=sim_config,
        n_histories=args.n_histories_val,
        soc_range=soc_range,
        seed=args.seed_val,
        output_path=output_dir / "val.npz",
    )
    print(f"  Saved to {val_path}")

    # --- Test data ---
    print(
        f"Generating {args.n_test_cases} test case(s) "
        f"(n_simu={args.n_simu}, n_intermediate_socs={args.n_intermediate_socs}, "
        f"seed={args.seed_test})..."
    )
    test_paths = generate_test_data(
        simulator_config=sim_config,
        n_test_cases=args.n_test_cases,
        n_simu=args.n_simu,
        n_intermediate_socs=args.n_intermediate_socs,
        seed=args.seed_test,
        output_dir=output_dir,
    )
    for tp in test_paths:
        print(f"  Saved to {tp}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
