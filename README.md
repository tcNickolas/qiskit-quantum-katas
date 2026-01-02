# Qiskit Quantum Katas

This project offers a Qiskit translation of the [Quantum Katas](https://quantum.microsoft.com/en-us/tools/quantum-katas).

## Introduction

The Quantum Katas are a collection of self-paced tutorials and programming exercises to help you learn quantum computing and quantum programming.
Each kata covers one quantum computing topic, introducing you to a subset of the basic concepts and algorithms used in quantum computing. The theoretical concepts are supported with programming exercises.

### Setup

* Python 3.12
* Visual Studio Code + Jupyter extension
* qiskit 2.2.3
* qiskit-aer 0.17.2

### Using the Quantum Katas

Each kata consists of three files: 

* The Jupyter notebook "frontend" offers you the description of the theoretical concepts and/or algorithms introduced in the kata and a set of programming exercises. This is the primary file you're work with when going through the kata.
* The Python "backend" contains a testing framework that sets up, runs, and validates your solutions to the programming exercises.
* The Jupyter notebook "workbook" offers explanations and code implementations of solutions to the programming exercises.

Each programming exercise gives you a task description and a signature of a Python function. Your goal is to implement the function that solves the task.
Once you've done that, execute the Jupyter Notebook code cell with yours solution to check whether it is correct. Once your code passes the test, you can move on to the next task!

The sequence of exercises in each kata progresses from easy to hard. The first exercise might require just one line of code or even a single gate, and the later ones might require rather complicated code.

### Testing reference solutions

You can run the tests on the solutions provided in the workbook(s) by running `pytest` in the root folder of the katas project or in a folder that contains a specific kata. You'll need additional packages `pytest` and `importnb`.

## Learning path

### Quantum computing concepts: qubits and gates

* [The qubit](./Qubit/Qubit.ipynb). Learn what a qubit is.
* [Single-qubit gates](./SingleQubitGates/SingleQubitGates.ipynb). Learn about what quantum gate is and about the most common single-qubit gates.
* [Multi-qubit systems](./MultiQubitSystems/MultiQubitSystems.ipynb). Learn how to represent multi-qubit systems.
* [Multi-qubit gates](./MultiQubitGates/MultiQubitGates.ipynb). Learn how to use multi-qubit gates and controlled variants of gates.
* [Preparing quantum states](./PreparingStates/PreparingStates.ipynb). Learn to prepare superposition states.

### Quantum computing concepts: measurements

* [Measurements in single-qubit systems](./SingleQubitMeasurements/SingleQubitMeasurements.ipynb). Learn about what quantum measurement is and how to use it in single-qubit systems.
* [Measurements in multi-qubit systems](./MultiQubitMeasurements/MultiQubitMeasurements.ipynb). Learn to use measurements in multi-qubit systems.
* [Distinguishing quantum states](./DistinguishingStates/DistinguishingStates.ipynb). Learn to distinguish orthogonal quantum states using measurements.

### Quantum oracles and simple oracle algorithms

* [Deutsch algorithm](./DeutschAlgorithm/DeutschAlgorithm.ipynb). Learn to implement single-qubit quantum oracles and compare the quantum solution to the Deutsch problem to a classical one.
* [Deutsch–Jozsa algorithm](./DeutschJozsaAlgorithm/DeutschJozsaAlgorithm.ipynb). Learn about quantum oracles which implement classical functions, and implement Bernstein–Vazirani and Deutsch–Jozsa algorithms.

