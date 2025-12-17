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
            import Workbook_DeutschAlgorithm as ref
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


def test_oracle_one_minus_x(fun=ref.oracle_one_minus_x if ref_available else None):
    expected_matrix = [[-1, 0], [0, 1]]
    check_operation_matrix(fun, expected_matrix)


def test_is_function_constant(fun=ref.is_function_constant if ref_available else None):
    def oracle_zero(circ: QuantumCircuit, q: QuantumRegister) -> None:
        ...

    def oracle_one(circ: QuantumCircuit, q: QuantumRegister) -> None:
        circ.x(q)
        circ.z(q)
        circ.x(q)
        circ.z(q)

    def oracle_x(circ: QuantumCircuit, q: QuantumRegister) -> None:
        circ.z(q)

    def oracle_one_minus_x(circ: QuantumCircuit, q: QuantumRegister) -> None:
        circ.x(q)
        circ.z(q)
        circ.x(q)

    def function_type(type: bool) -> str:
        return "constant" if type else "variable"

    for (oracle, expected, name) in [
        (oracle_zero, True, 'f(x) = 0'),
        (oracle_one, True, 'f(x) = 1'),
        (oracle_x, False, 'f(x) = x'),
        (oracle_one_minus_x, False, 'f(x) = 1 - x')
    ]:
        actual = fun(oracle)
        assert actual == expected, f"{name} identified as {function_type(actual)} but it is {function_type(expected)}"
