from cmath import exp
from functools import partial
from math import pi, sqrt, cos, sin
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
            import Workbook_MultiQubitGates as ref
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


def print_matrix(matrix):
    for row in matrix:
        print(row)


def check_operation_matrix(
    n,              # Number of qubits
    solution,       # Callable that is being tested
    expected_matrix # Matrix it should have
) -> None:
    # Construct the circuit that has the callable as a part of it
    qr = QuantumRegister(n)
    circ = QuantumCircuit(qr)
    solution(circ, qr)
    # Convert the circuit to a matrix
    op = Operator(circ)
    actual_matrix = op.data

    # Check that the actual matrix matches the expected one
    for actual, expected in zip(actual_matrix, expected_matrix):
        if actual != approx(expected):
            print("Expected matrix:")
            print_matrix(expected_matrix)
            print("Actual matrix:")
            print_matrix(actual_matrix)
            raise ValueError("Operation matrices should be equal")


def test_apply_tensor_product(fun=ref.apply_tensor_product if ref_available else None):
    expected_matrix = [[0, -1j, 0, 0, 0, 0, 0, 0],
                       [1j,  0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1j, 0, 0, 0, 0],
                       [0, 0, 1j,  0, 0, 0, 0, 0],
                       [0, 0, 0, 0,  0, 1, 0, 0],
                       [0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0,  0, 1],
                       [0, 0, 0, 0, 0, 0, -1, 0],
                       ]
    check_operation_matrix(3, fun, expected_matrix)


isq2 = 1 / sqrt(2)

def test_prepare_bell_state(fun=ref.prepare_bell_state if ref_available else None):
    expected_vector = [isq2, 0, 0, isq2]
    check_state_vector(fun, 2, expected_vector)


def test_entangle_two_qubits(fun=ref.entangle_two_qubits if ref_available else None):
    def initial_state_prep(circ, qr):
        circ.h(qr)

    expected_vector = [0.5, 0.5, 0.5, -0.5]
    check_state_vector(fun, 2, expected_vector, initial_state_prep)


def test_swap_amplitudes(fun=ref.swap_amplitudes if ref_available else None):
    expected_matrix = [[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       ]
    check_operation_matrix(2, fun, expected_matrix)


def test_fredkin_gate(fun=ref.fredkin_gate if ref_available else None):
    # Swap basis states 3 and 5
    expected_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       ]
    check_operation_matrix(3, fun, expected_matrix)


def test_controlled_rotation(fun=ref.controlled_rotation if ref_available else None):
    for i in range(20):
        theta = pi * i / 10
        expected_matrix = [[1, 0, 0, 0],
                           [0, cos(theta/2), 0, -sin(theta/2)],
                           [0, 0, 1, 0],
                           [0, sin(theta/2), 0, cos(theta/2)],
                           ]
        fun_theta = partial(fun, theta=theta)
        check_operation_matrix(2, fun_theta, expected_matrix)


def test_controlled_phase(fun=ref.controlled_phase if ref_available else None):
    for i in range(20):
        theta = pi * i / 10
        expected_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, exp(1j * theta)],
                           ]
        fun_theta = partial(fun, theta=theta)
        check_operation_matrix(3, fun_theta, expected_matrix)


def test_anti_controlled_gate(fun=ref.anti_controlled_gate if ref_available else None):
    expected_matrix = [[0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1],
                       ]
    check_operation_matrix(2, fun, expected_matrix)
