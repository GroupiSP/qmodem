---
name: qavi_rul
description: estimate the remaining useful life (RUL) of (simulated) batteries with a Bayesian (C)NN trained with quantum adversarial variational inference (QAVI).
---

- #file:../../sandbox/qavi_regression_harmonics/ contains the details of the QAVI implementation for a regression problem.
- #file:../../scripts/bayes_cnn/ contains the whole procedure that should be followed also to implement QAVI for RUL estimation. The task at hand is the same as the one performed in this folder, but this time the CNN kernel weights and biases should be sampled from the PQC generator, rather than from a Gaussian distribution.

# Implementation details
- Use 4 PQC generators, one for each filter of the convolutional layer.
- Each PQC generator should have 6 qubits. The expectation values of the first 5 qubits should be used to sample the kernel weights, and the expectation value of the 6th qubit should be used to sample the bias.
- Each PQC generator should have its own post-processor, which should be just a linear layer.
