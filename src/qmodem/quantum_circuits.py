from __future__ import annotations

from typing import Protocol

import jax
import numpy as np
import pennylane as qp

from .pennylane_typing import Device, ExpectationMP, SampleMP


class CircuitFactory(Protocol):
    @property
    def params_example(self) -> np.ndarray: ...

    def make_circuit(self, device: Device, n_shots: int | None) -> qp.QNode: ...


class ContinuousCircuitFactory:
    def __init__(self, n_qubits: int, n_layers: int):
        """Factory for parameterized quantum circuits (PQC) that generate a quantum
        state based on the input x and parameters.

        x is assumed to be a sample from a given distribution (e.g., uniform or normal)
        and is used to encode the input into the quantum state.
        """
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers

    @property
    def params_example(self) -> np.ndarray:
        """Return the shape of the parameters for the circuit."""
        return np.random.normal(size=(self.n_layers, self.n_qubits, 2))

    def make_circuit(self, device: Device, n_shots: int | None = None) -> qp.QNode:
        """Create a parameterized quantum circuit (PQC) that generates a quantum state
        based on the input x and parameters.

        The final measurement is taken on the computational basis of the qubits.

        Args:
            device (Device): The quantum device to use for the circuit.
            n_shots (int | None): The number of shots to use for the circuit. Ignored for this circuit,
                as it uses the statevector simulator. Defaults to None.
        """

        @qp.qnode(device=device, interface="jax")
        def circuit(x: jax.Array, params: jax.Array) -> list[ExpectationMP]:
            for q in range(self.n_qubits):
                qp.RY(x, wires=q)
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qp.RY(params[layer, q, 0], wires=q)
                    qp.RZ(params[layer, q, 1], wires=q)
                for q in range(self.n_qubits):
                    qp.CNOT(wires=[q, (q + 1) % self.n_qubits])
            return [qp.expval(qp.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit


class BinaryCircuitFactory:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers

    @property
    def params_example(self) -> np.ndarray:
        """Return the shape of the parameters for the circuit."""
        return np.random.normal(size=(self.n_layers, self.n_qubits, 2))

    def make_circuit(self, device: Device, n_shots: int = 1) -> qp.QNode:
        """Factory for Quantum Circuit Born Machines (QCBMs).

        The final measurement is taken on the computational basis of the qubits.
        """

        @qp.set_shots(n_shots)
        @qp.qnode(device=device, interface="jax")
        def circuit(params: jax.Array) -> SampleMP:
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qp.RY(params[layer, q, 0], wires=q)
                    qp.RZ(params[layer, q, 1], wires=q)
                for q in range(self.n_qubits):
                    qp.CNOT(wires=[q, (q + 1) % self.n_qubits])
            return qp.sample()  # Sample from the computational basis of the qubits

        return circuit
