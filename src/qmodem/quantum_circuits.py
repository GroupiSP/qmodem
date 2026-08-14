from __future__ import annotations

from typing import TYPE_CHECKING

import pennylane as qp


if TYPE_CHECKING:
    from pennylane.devices import Device
    from pennylane.measurements import ExpectationMP, SampleMP
    from pennylane.typing import TensorLike


def continuous_circuit_factory(device: Device, n_layers: int = 1) -> qp.QNode:
    """Factory for parameterized quantum circuits (PQC) that generate a quantum state
    based on the input x and parameters.

    x is assumed to be a sample from a given distribution (e.g., uniform or normal) and
    is used to encode the input into the quantum state.
    """
    n_qubits = len(device.wires)

    @qp.qnode(device=device, interface="jax")
    def circuit(x: TensorLike, params: TensorLike) -> list[ExpectationMP]:
        for q in range(n_qubits):
            qp.RY(x, wires=q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qp.RY(params[layer, q, 0], wires=q)
                qp.RZ(params[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qp.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def binary_circuit_factory(
    device: Device, n_shots: int = 1, n_layers: int = 1
) -> qp.QNode:
    """Factory for Quantum Circuit Born Machines (QCBMs).

    The final measurement is taken on the computational basis of the qubits.
    """
    n_qubits = len(device.wires)

    @qp.set_shots(n_shots)
    @qp.qnode(device=device, interface="jax")
    def circuit(params: TensorLike) -> SampleMP:
        for layer in range(n_layers):
            for q in range(n_qubits):
                qp.RY(params[layer, q, 0], wires=q)
                qp.RZ(params[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qp.CNOT(wires=[q, (q + 1) % n_qubits])
        return qp.sample()  # Sample from the computational basis of the qubits

    return circuit
