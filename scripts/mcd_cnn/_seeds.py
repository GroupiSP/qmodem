"""Shared random seeds for reproducibility and train/test isolation.

Training and test scripts use different seeds to guarantee independently sampled
discharge histories from the same distribution. All test/evaluation scripts share the
same TEST_SEED so they evaluate on the same observed trajectory.
"""

TRAIN_SEED = 42
TEST_SEED = 123
