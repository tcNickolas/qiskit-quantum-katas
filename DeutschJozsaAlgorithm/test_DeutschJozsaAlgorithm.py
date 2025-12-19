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
            import Workbook_DeutschJozsaAlgorithm as ref
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
    n,              # Number of qubits
    solution,       # Callable that is being tested
    expected_matrix # Matrix it should have
) -> None:
    # Construct the circuit that has the callable as a part of it
    q = QuantumRegister(n)
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


def test_oracle_msb_x(fun=ref.oracle_msb_x if ref_available else None):
    for n in range(1, 5):
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        for i in range(2 ** n):
            expected_matrix[i][i] = 1 if i & (2 ** (n - 1)) == 0 else -1
        check_operation_matrix(n, fun, expected_matrix)


def test_oracle_parity(fun=ref.oracle_parity if ref_available else None):
    for n in range(1, 5):
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        for i in range(2 ** n):
            expected_matrix[i][i] = -1 if i.bit_count() % 2 == 1 else 1
        check_operation_matrix(n, fun, expected_matrix)



def oracle_zero(circ: QuantumCircuit, qr: QuantumRegister) -> None:
    ...

def oracle_one(circ: QuantumCircuit, qr: QuantumRegister) -> None:
    circ.x(qr[0])
    circ.z(qr[0])
    circ.x(qr[0])
    circ.z(qr[0])

def oracle_x_mod_2(circ: QuantumCircuit, qr: QuantumRegister) -> None:
    circ.z(qr[0])

def oracle_middle_bit(circ: QuantumCircuit, qr: QuantumRegister) -> None:
    circ.z(qr[1])

def oracle_msb_x(circ: QuantumCircuit, qr: QuantumRegister) -> None:
    circ.z(qr[-1])

def oracle_parity(circ: QuantumCircuit, qr: QuantumRegister) -> None:
    circ.z(qr)

def test_is_function_constant(fun=ref.is_function_constant if ref_available else None):
    def function_type(type: bool) -> str:
        return "constant" if type else "balanced"

    for (oracle, expected, name) in [
        (oracle_zero, True, 'f(x) = 0'),
        (oracle_one, True, 'f(x) = 1'),
        (oracle_x_mod_2, False, 'f(x) = x mod 2'),
        (oracle_msb_x, False, 'f(x) = MSB(x)'),
        (oracle_parity, False, 'f(x) = PARITY(x)')
    ]:
        for n in range(1, 5):
            actual = fun(n, oracle)
            assert actual == expected, f"{name} for {n=} identified as {function_type(actual)} but it is {function_type(expected)}"


def test_bernstein_vazirani_algorithm(fun=ref.bernstein_vazirani_algorithm if ref_available else None):
    for (n, oracle, expected, name) in [
        (2, oracle_zero, [0, 0], 'f(x) = 0'),
        (3, oracle_zero, [0, 0, 0], 'f(x) = 0'),
        (2, oracle_parity, [1, 1], 'f(x) = PARITY(x)'),
        (3, oracle_parity, [1, 1, 1], 'f(x) = PARITY(x)'),
        (2, oracle_x_mod_2, [1, 0], 'f(x) = x mod 2'),
        (3, oracle_x_mod_2, [1, 0, 0], 'f(x) = x mod 2'),
        (2, oracle_msb_x, [0, 1], 'f(x) = MSB(x)'),
        (3, oracle_msb_x, [0, 0, 1], 'f(x) = MSB(x)'),
        (3, oracle_middle_bit, [0, 1, 0], 'f(x) = middle bit of x')
    ]:
        actual = fun(n, oracle)
        assert actual == expected, f"The bit string for {name} for {n=} identified as {actual} but it is {expected}"
