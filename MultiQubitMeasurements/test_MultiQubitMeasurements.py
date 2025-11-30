from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_MultiQubitMeasurements as ref
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
            # Check that the measured state matches the state that was prepared
            res_ind = int(list(res_counts.keys())[0], 2)
            if res_ind != ind:
                raise ValueError(f"Unexpected measurement outcome: expected {ind}, got {res_ind}")
        except Exception as e:
            raise Exception(f"Testing on state {state_names[ind]}: {e}")


def test_measure_basis_state(solution=ref.measure_basis_state if ref_available else None):
    amps = [[1, 0, 0, 0], 
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
    check_distinguish_states(2, 4, ["|00⟩", "|10⟩", "|01⟩", "|11⟩"], amps, solution)


def test_measure_plusminus_state(solution=ref.measure_plusminus_state if ref_available else None):
    amps = [[1, 1, -1, -1], 
            [1, -1, -1, 1]]
    check_distinguish_states(2, 2, ["|+-⟩", "|--⟩"], amps, solution)
