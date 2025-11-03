from cmath import exp
from math import pi, sqrt, cos, sin
from functools import partial
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_SingleQubitMeasurements as ref
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


def check_distinguish_states(
    n_qubits,    # Number of qubits in the register
    n_states,    # Number of different states
    state_names, # Readable names of the states
    state_amps,  # Amplitudes of each state
    solution     # Callable that is being tested
):
    n_shots = 100

    for ind in range(n_states):
        # print(f"Running experiments on state {state_names[ind]}...")
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        circ = QuantumCircuit(qr, cr)

        # Prepare the state number ind
        circ.initialize(state_amps[ind], qr, normalize=True)

        try:
            # Apply the callable
            solution(circ, qr, cr)

            # Run the simulation
            res_counts = simulator.run(circ).result().get_counts()

            # Interpret the results
            # Check that the execution result is always the same (we're only looking at deterministic measurements here)
            if len(res_counts) > 1:
                raise ValueError(f"Non-deterministic measurement outcome: {res_counts}")
            # Check that the measured state matches the state that was prepared - in little-endian
            res_ind = int(list(res_counts.keys())[0][::-1], 2)
            if res_ind != ind:
                raise ValueError(f"Unexpected measurement outcome: expected {ind}, got {res_ind}")
        except Exception as e:
            raise Exception(f"Testing on state {state_names[ind]}: {e}")


def test_zero_or_one(solution=ref.zero_or_one if ref_available else None):
    amps = [[1, 0], 
            [0, 1]]
    check_distinguish_states(1, 2, ["|0⟩", "|1⟩"], amps, solution)


def test_plus_or_minus(solution=ref.plus_or_minus if ref_available else None):
    amps = [[1, 1], 
            [1, -1]]
    check_distinguish_states(1, 2, ["|+⟩", "|-⟩"], amps, solution)


def test_psi_plus_or_psi_minus(solution=ref.psi_plus_or_psi_minus if ref_available else None):
    amps = [[0.6, 0.8], 
            [-0.8, 0.6]]
    check_distinguish_states(1, 2, ["|Ψ+⟩", "|Ψ-⟩"], amps, solution)


def test_a_or_b(solution=ref.a_or_b if ref_available else None):
    from math import cos, sin, pi
    for i in range(11):
        alpha = i * pi / 10
        amps = [[cos(alpha), -1j * sin(alpha)], 
                [-1j * sin(alpha), cos(alpha)]]
        names = [f"|A⟩ = cos({i}π/10)|0⟩ - i sin({i}π/10)|1⟩", 
                 f"|B⟩ = -i sin({i}π/10)|0⟩ + cos({i}π/10)|1⟩"]
        sol = partial(solution, alpha=alpha)
        check_distinguish_states(1, 2, names, amps, sol)
