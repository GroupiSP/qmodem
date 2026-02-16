# Runnable scripts
This directory contains a set of runnable scripts for benchmark and quantum-aided prognostic models.

## Design of experiments
The data for all scripts is obtained from the battery simulator (`SimulatorSimple`) available in `lib_eod_sim`. The input data always consists of the discharge voltage, while the label is the remaining useful life (RUL).

__Note__: the labels are assigned using a __linear degradation model__.

__Note__: the battery simulator in `lib_eod_sim` is a noisy battery simulator. It can simulate process noise and observation noise. The first is a Gaussian noise on the state-of-charge (SoC) updates, while the second one simulates misreadings of the terminal voltage.

### Training simulations
The training data is extracted by the discharge histories simulated from different initial SoC. The initial SoC is chosen uniformly at random for each discharge, in order to cover the scenario of batteries discharging from multiple levels of charge.

### Testing simulations
The test phase consists of two parts.
1. RUL prediction and confidence intervals.
2. CRPS over time.

For both parts, a single battery discharge is simulated, sampled from the same simulator distribution as used in training, to satisfy IID-ness.

#### RUL prediction and uncertainty
The discharge voltage from the test history is fed as input to the data-driven model (DDM). A first deterministic comparison is obtained by the plotting the "true" RUL modelled with the linear degradation and the deterministic prediction of the DDM.

A second comparison plots the uncertainty intervals as well. For the simulated data, the RUL uncertainty over time is obtained by running `SimulatorSimple` with the original configuration, but from initial SoCs along the test history and for multiple runs. This creates RUL distributions over time, which represent the _aleatoric uncertainty_ of the problem.

#### CRPS over time
The true distribution used in the CRPS calculation is obtained in the same way as done for the true RUL uncertainty quantification.
