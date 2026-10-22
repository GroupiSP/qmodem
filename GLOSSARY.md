# qmodem

The shared vocabulary of qmodem: quantum-enhanced uncertainty quantification for remaining-useful-life regression, and the protocol that compares quantum and classical inference methods. Training-run terms (stage, posterior stage, inference method, posterior, draw, aleatoric/epistemic variance) are stagereg's and are defined in the [stagereg glossary](https://github.com/GroupiSP/stagereg/blob/main/GLOSSARY.md); they are not redefined here.

## Inference methods

**AVI**:
Adversarial variational inference: the family of inference methods whose approximate posterior over part of the network's weights is implicit, defined by a generator, and fitted adversarially against a discriminator.
_Avoid_: AVB (fits a weight-space discriminator), CAVI (coordinate-ascent VI)

**Q-AVI**:
The AVI method whose generator has a quantum latent map.
_Avoid_: QAVI, Quantum BNN, quantum hypernetwork

**C-AVI**:
The AVI method whose generator has a classical latent map; the ablation arm that isolates Q-AVI's quantum circuit.
_Avoid_: CAVI, classical QAVI

**Generator**:
A latent map followed by a postprocessor, mapping a random input to one draw of the inferred weights.
_Avoid_: Weights generator, hypernetwork

**Latent map**:
The part of a generator that maps the random input to a bounded feature vector: a parametrized quantum circuit and its measurements (quantum latent map), or a classical network matched to one in output width, input and parameter count (classical latent map).
_Avoid_: Feature map (in QML, only the data-encoding part of a circuit)

**Postprocessor**:
The classical map that turns a latent map's output into weights of the inferred part of the network.
_Avoid_: Readout, decoder

**Discriminator**:
A classical network, external to the regression network, that judges whether an input–target pair is real or produced by the regression network; it exists only to train AVI methods.
_Avoid_: Critic

**Adversarial weight**:
The per-example trade-off, in an AVI method's objective, between fooling the discriminator and fitting the likelihood and prior.
_Avoid_: GAN weight, adversarial loss weight

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
The run that restores a backbone run and adds one inference method's posterior stage (none for HNN), and that holds that method's evaluation. Its method is named by one tag value: `hnn` (HNN), `mcd` (MCD), `bbb` (BBB), `psgld` (pSGLD), `q-avi` (Q-AVI) or `c-avi` (C-AVI).
_Avoid_: Child run, posterior run, evaluation run

**Protocol seed**:
The index that fixes one backbone run and the method runs restored from it; methods are compared paired by protocol seed.
_Avoid_: Repetition, trial (a trial is an HPO term)

**HPO study**:
The search that fixes one stage's hyperparameters on protocol seed 0 (the mean and variance stages once for all methods, the posterior stage once per method), made of trials whose winning hyperparameters are then frozen for every protocol seed.
_Avoid_: Tuning run, sweep

**Backbone validation set**:
The held-out histories that select, stop and tune the mean and variance stages; disjoint from the posterior validation set, so that no posterior stage is judged on data its backbone was selected on.
_Avoid_: val_A, holdout

**Posterior validation set**:
The held-out histories, disjoint from the backbone validation set, on which the validation criterion is computed.
_Avoid_: val_B, holdout

**Validation timestamps**:
The fixed, evenly spaced points of each history of a validation set at which every validation score is computed, the same for every epoch, trial and method; the validation counterpart of a test case's evaluation timestamps.
_Avoid_: Validation windows (suggests every window), validation subsample

**Validation criterion**:
The single score, at the validation timestamps of the posterior validation set, that selects a posterior stage's best checkpoint, stops it early and serves as its HPO objective, shared by every method of the comparison protocol.
_Avoid_: Validation loss (training losses differ by method), monitored metric (the stopper's mechanism, not the protocol's choice)

**Ablation arm**:
An optional set of method runs that changes one component of a core method, reported against the core methods and never ranked among them.
_Avoid_: Variant, baseline

**Sensitivity arm**:
An optional set of method runs that changes one setting of the posterior problem shared by all methods, reported against the primary setting.
_Avoid_: Ablation (an ablation changes a method, not the shared problem)

## Outputs

The reported results of the comparison protocol, named by a fixed ID. T1, T2, F1 and F2 are the primary results.

**T1**:
The ranking table: RMSE, CRPS, coverage@0.95 and WSU per core method (mean ± std across protocol seeds), overall and per policy.
_Avoid_: Results table, main table

**T2**:
The paired-difference table: each method's per-seed difference to BBB and to HNN, with a bootstrap confidence interval over protocol seeds; ablation and sensitivity arms appear here, one row each.
_Avoid_: Delta table, significance table

**T3**:
The cost table: the posterior stage's wall-clock time, trainable parameter count and posterior draws per second, per method.
_Avoid_: Timing table, runtime table

**F1**:
The reliability figure: empirical against nominal coverage of the predictive intervals at levels 0.5 to 0.95, per method.
_Avoid_: Calibration plot

**F2**:
The variance-decomposition figure: epistemic and aleatoric variance over initial state of charge, per method, on the representative case of each current policy.
_Avoid_: Uncertainty plot

**F3**:
The per-case figure: CRPS of every test case for every method, grouped by policy.
_Avoid_: Per-trajectory bars

**F4**:
The RUL-band figure: each method's 95% predictive band against the simulator's reference band, on the representative case of each current policy.
_Avoid_: RUL plot, prediction plot

**Representative case**:
For one current policy, the test case whose HNN CRPS is the median among that policy's test cases at protocol seed 0, fixed before any method results are seen.
_Avoid_: Example case, typical trajectory
