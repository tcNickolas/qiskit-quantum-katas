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
            import Workbook_SingleQubitGates as ref
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
    solution,       # Callable that is being tested
    expected_vector # State vector it should prepare
) -> None:
    # Construct the circuit that has the callable as a part of it
    q = QuantumRegister(1)
    circ = QuantumCircuit(q)
    solution(circ, q)

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
    solution,       # Callable that is being tested
    expected_matrix # Matrix it should have
) -> None:
    # Construct the circuit that has the callable as a part of it
    q = QuantumRegister(1)
    circ = QuantumCircuit(q)
    solution(circ, q)
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


def test_state_flip(fun=ref.state_flip if ref_available else None):
    expected_matrix = [[0, 1], [1, 0]]
    check_operation_matrix(fun, expected_matrix)


def test_sign_flip(fun=ref.sign_flip if ref_available else None):
    expected_matrix = [[1, 0], [0, -1]]
    check_operation_matrix(fun, expected_matrix)


def test_apply_y(fun=ref.apply_y if ref_available else None):
    expected_matrix = [[0, -1j], [1j, 0]]
    check_operation_matrix(fun, expected_matrix)


def test_sign_flip_on_zero(fun=ref.sign_flip_on_zero if ref_available else None):
    expected_matrix = [[-1, 0], [0, 1]]
    check_operation_matrix(fun, expected_matrix)


def test_global_phase_minus_one(fun=ref.global_phase_minus_one if ref_available else None):
    expected_matrix = [[-1, 0], [0, -1]]
    check_operation_matrix(fun, expected_matrix)


def test_global_phase_i(fun=ref.global_phase_i if ref_available else None):
    expected_matrix = [[1j, 0], [0, 1j]]
    check_operation_matrix(fun, expected_matrix)


def test_basis_change(fun=ref.basis_change if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_matrix = [[sqrt12, sqrt12], [sqrt12, -sqrt12]]
    check_operation_matrix(fun, expected_matrix)


def test_prepare_plus(fun=ref.prepare_plus if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, sqrt12]
    check_state_vector(fun, expected_vector)


def test_prepare_minus(fun=ref.prepare_minus if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, -sqrt12]
    check_state_vector(fun, expected_vector)


def test_relative_phase_i(fun=ref.relative_phase_i if ref_available else None):
    expected_matrix = [[1, 0], [0, 1j]]
    check_operation_matrix(fun, expected_matrix)


def test_relative_phase_three_quarters_pi(fun=ref.relative_phase_three_quarters_pi if ref_available else None):
    expected_matrix = [[1, 0], [0, exp(3 * pi / 4 * 1j)]]
    check_operation_matrix(fun, expected_matrix)


def test_amplitude_change(fun=ref.amplitude_change if ref_available else None):
    for ind in range(37):
        gamma = 2 * pi * ind / 36
        expected_matrix = [[cos(gamma), -sin(gamma)], [sin(gamma), cos(gamma)]]
        fun_gamma = partial(fun, gamma=gamma)
        check_operation_matrix(fun_gamma, expected_matrix)


def test_relative_phase_change(fun=ref.relative_phase_change if ref_available else None):
    for ind in range(37):
        gamma = 2 * pi * ind / 36
        expected_matrix = [[1, 0], [0, exp(1j * gamma)]]
        fun_gamma = partial(fun, gamma=gamma)
        check_operation_matrix(fun_gamma, expected_matrix)


def test_prepare_rotated_state(fun=ref.prepare_rotated_state if ref_available else None):
    for ind in range(11):
        alpha = cos(ind)
        beta = sin(ind)
        expected_vector = [alpha, -1j * beta]
        fun_alpha_beta = partial(fun, alpha=alpha, beta=beta)
        check_state_vector(fun_alpha_beta, expected_vector)


def test_prepare_arbitrary_state(fun=ref.prepare_arbitrary_state if ref_available else None):
    for ind in range(11):
        alpha = cos(ind)
        beta = sin(ind)
        theta = sin(ind * 3)
        expected_vector = [alpha, exp(1j * theta) * beta]
        fun_alpha_beta = partial(fun, alpha=alpha, beta=beta, theta=theta)
        check_state_vector(fun_alpha_beta, expected_vector)
