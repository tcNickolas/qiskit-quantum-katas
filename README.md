# Qiskit Quantum Katas

Qiskit translation of the [Quantum Katas](https://quantum.microsoft.com/en-us/tools/quantum-katas).

## Introduction

The Quantum Katas are a collection of self-paced tutorials and programming exercises to help you learn quantum computing and quantum programming.

* Each kata covers one quantum computing topic, introducing you to the basic concepts and algorithms used in quantum computing. The theoretical concepts are supported with programming exercises.
* Each exercise gives you a task description and a signature of a Python function. Your goal is to implement the function that solves the task.
* The sequence of exercises progresses from easy to hard. The first exercise might require just one line of code or even a single gate, and the later ones might require rather complicated code.
* Each kata has a testing framework that sets up, runs, and validates your solutions. Execute the Jupyter Notebook code cell with yours solution to check whether it is correct. Once your code passes the test, you can move on to the next task.


## Learning path

### Quantum computing concepts: qubits and gates

* [The qubit](./Qubit/Qubit.ipynb). Learn what a qubit is.
* [Single-qubit gates](./SingleQubitGates/SingleQubitGates.ipynb). Learn about what quantum gate is and about the most common single-qubit gates.
* [Multi-qubit systems](./MultiQubitSystems/MultiQubitSystems.ipynb). Learn how to represent multi-qubit systems.
* [Preparing quantum states](./PreparingStates/PreparingStates.ipynb). Learn to prepare superposition states.

### Quantum computing concepts: measurements

* [Measurements in single-qubit systems](./SingleQubitMeasurements/SingleQubitMeasurements.ipynb). Learn about what quantum measurement is and how to use it in single-qubit systems.
* [Measurements in multi-qubit systems](./MultiQubitMeasurements/MultiQubitMeasurements.ipynb). Learn to use measurements in multi-qubit systems.
* [Distinguishing quantum states](./DistinguishingStates/DistinguishingStates.ipynb). Learn to distinguish orthogonal quantum states using measurements.

### Quantum oracles and simple oracle algorithms

* [Deutsch algorithm](./DeutschAlgorithm/DeutschAlgorithm.ipynb). Learn to implement single-qubit quantum oracles and compare the quantum solution to the Deutsch problem to a classical one.
* [Deutsch–Jozsa algorithm](./DeutschJozsaAlgorithm/DeutschJozsaAlgorithm.ipynb). Learn about quantum oracles which implement classical functions, and implement Bernstein–Vazirani and Deutsch–Jozsa algorithms.

