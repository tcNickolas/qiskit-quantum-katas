from cmath import exp
from math import pi, sqrt, cos, sin
from functools import partial
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_MultiQubitSystems as ref
    ref_available = True
except ImportError:
    ref_available = False
    # Skip all tests in this file - pytest checks reference solutions and that won't work without these imports
    pytestmark = mark.skip("No importnb/reference file available")


def exercise(fun):
    test_name = "test_" + fun.__name__
    try:
        test_func = globals()[test_name]
    except KeyError:
        print(f"Test {test_name} not found")
    else:
        try:
            test_func(fun)
        except Exception as e:
            print("Incorrect")
            print(e)
        else:
            print("Correct!")
    
    return fun


# Create the simulator instance to add save_statevector method to QuantumCircuit
simulator = AerSimulator(method='statevector')


def check_state_vector(
    solution,        # Callable that is being tested
    n_qubits,        # Number of qubits in the register
    expected_vector, # State vector it should prepare
    initial_state_prep = None  # A routine that prepares the initial state (before the solution is called)
) -> None:
    # Construct the circuit that has the callable as a part of it
    qr = QuantumRegister(n_qubits)
    circ = QuantumCircuit(qr)

    if initial_state_prep is not None:
        initial_state_prep(circ, qr)

    solution(circ, qr)

    # Save the state vector
    circ.save_statevector()

    # Run the simulation and get the results
    res = simulator.run(circ).result()
    # Extract the saved state vector
    actual_vector = res.get_statevector().data

    if actual_vector != approx(expected_vector):
        print("Expected state vector:")
        print(expected_vector)
        print("Actual state vector:")
        print(actual_vector)
        raise ValueError("State vectors should be equal")


def test_prepare_oneone(fun=ref.prepare_oneone if ref_available else None):
    expected_vector = [0, 0, 0, 1]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_superposition(fun=ref.prepare_superposition if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, 0, -sqrt12, 0]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_real_amplitudes(fun=ref.prepare_real_amplitudes if ref_available else None):
    expected_vector = [0.5, 0.5, -0.5, -0.5]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_complex_amplitudes(fun=ref.prepare_complex_amplitudes if ref_available else None):
    expected_vector = [0.5, 0.5 * exp(1j * pi / 2), 0.5 * exp(1j * pi / 4), 0.5 * exp(1j * 3 * pi / 4)]
    check_state_vector(fun, 2, expected_vector)


def bell_state_prep(circ: QuantumCircuit, qr: QuantumRegister):
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])


def test_prepare_bell_state_1(fun=ref.prepare_bell_state_1 if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, 0, 0, -sqrt12]
    check_state_vector(fun, 2, expected_vector, bell_state_prep)


def test_prepare_bell_state_2(fun=ref.prepare_bell_state_2 if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [0, sqrt12, sqrt12, 0]
    check_state_vector(fun, 2, expected_vector, bell_state_prep)


def test_prepare_bell_state_3(fun=ref.prepare_bell_state_3 if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [0, -sqrt12, sqrt12, 0]
    check_state_vector(fun, 2, expected_vector, bell_state_prep)
