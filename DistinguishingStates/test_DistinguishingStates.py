from cmath import exp
from math import pi, sqrt, cos, sin, log2
from functools import partial
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_DistinguishingStates as ref
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
    solution,    # Callable that is being tested
    n_meas=None  # Number of measurements allowed (n_qubits if None)
):
    n_shots = 100

    for ind in range(n_states):
        # print(f"Running experiments on state {state_names[ind]}...")
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits if n_meas is None else n_meas)
        circ = QuantumCircuit(qr, cr)

        # Prepare the state number ind
        circ.initialize(state_amps[ind], qr, normalize=True)

        try:
            # Apply the solution and possibly use its return
            decoder = solution(circ, qr, cr)

            # Run the simulation
            circ = transpile(circ, backend=simulator)
            res_counts = simulator.run(circ).result().get_counts()

            # Interpret the results
            if decoder is None:
                # Convert the result into integer (using little-endian and remembering that Qiskit reverses measurement results to make them big-endian) 
                # - first bit of key is the least significant
                # Check that the execution result is always the same (we're only looking at deterministic measurements here)
                if len(res_counts) > 1:
                    raise ValueError(f"Non-deterministic measurement outcome: {res_counts}")
                # Check that the measured state matches the state that was prepared
                res_ind = int(list(res_counts.keys())[0], 2)
                if res_ind != ind:
                    raise ValueError(f"Unexpected measurement outcome: expected {ind}, got {res_ind}")
            else:
                # Use decoder for each of the measurement outcomes - all must give correct result
                for res_bitstring in res_counts.keys():
                    res_ind = decoder(res_bitstring)
                    if res_ind != ind:
                        raise ValueError(f"Unexpected measurement decoding: raw measurement {res_bitstring}, expected {ind}, got {res_ind}")

        except Exception as e:
            raise Exception(f"Testing on state {state_names[ind]}: {e}")


def test_zerozero_or_oneone(solution=ref.zerozero_or_oneone if ref_available else None):
    amps = [[1, 0, 0, 0], 
            [0, 0, 0, 1]]
    check_distinguish_states(2, 2, ["|00⟩", "|11⟩"], amps, solution)


def test_measure_basis_state(solution=ref.measure_basis_state if ref_available else None):
    amps = [[1, 0, 0, 0], 
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
    check_distinguish_states(2, 4, ["|00⟩", "|10⟩", "|01⟩", "|11⟩"], amps, solution)


def _bitstring_as_int_le(bits):
    return sum([2 ** i if bits[i] else 0 for i in range(len(bits))])


def test_measure_two_bitstrings(solution=ref.measure_two_bitstrings if ref_available else None):
    for b0, b1 in [
            ([False, True], [True, False]),
            ([True, True, False], [False, True, True]),
            ([False, True, True, False], [False, True, True, True]),
            ([True, False, False, False], [True, False, True, True])
        ]:
        n_qubits = len(b1)
        amps = [[0] * 2 ** n_qubits, [0] * 2 ** n_qubits]
        amps[0][_bitstring_as_int_le(b0)] = 1
        amps[1][_bitstring_as_int_le(b1)] = 1
        solution_bits = partial(solution, bits0=b0, bits1=b1)
        check_distinguish_states(n_qubits, 2, [f"{b0}", f"{b1}"], amps, solution_bits)


def _int_as_bitstring_le(bits, n_bits):
    return [(bits & (1 << ind)) > 0 for ind in range(n_bits)]


def _ints_as_amps(n_bits, ints):
    amps = [0] * 2 ** n_bits
    for i in ints:
        amps[i] = 1 / sqrt(n_bits)
    return amps


def test_measure_four_bitstring_superpositions(solution=ref.measure_four_bitstring_superpositions if ref_available else None):
    for n, ints0, ints1 in [
        (2, [2], [1]),                        # [10] vs [01]
        (2, [2, 1], [3, 0]),                  # [10,01] vs [11,00]
        (2, [2], [3, 0]),                     # [10] vs [11,00]
        (2, [1, 2], [3]),                     # [01,10] vs [11]
        (3, [5, 7], [2]),                     # [101,111] vs [010]
        (4, [15, 6], [0, 14]),                # [1111,0110] vs [0000,1110]
        (4, [15, 7], [0, 8, 10, 13]),         # [1111,0111] vs [0000,1000,1010,1101]
        (4, [13, 11, 7, 3], [2, 5]),          # [1101,1011,0111,0011] vs [0010,0101]
        (5, [30, 14, 10, 7], [1, 17, 21, 27]) # [11110,01110,01010,00111] vs [00001,10001,10101,11011]
    ]:
        bits0 = [_int_as_bitstring_le(int0, n) for int0 in ints0]
        bits1 = [_int_as_bitstring_le(int1, n) for int1 in ints1]
        amps = [_ints_as_amps(n, ints0), _ints_as_amps(n, ints1)]
        solution_bits = partial(solution, bits0=bits0, bits1=bits1)
        check_distinguish_states(n, 2, [f"{bits0}", f"{bits1}"], amps, solution_bits)


def test_measure_four_bitstring_superpositions_one(solution=ref.measure_four_bitstring_superpositions_one if ref_available else None):
    for n, ints0, ints1 in [
        (2, [2], [1]),                        # [10] vs [01]
        (2, [2, 3], [1, 0]),                  # [10,11] vs [01,00]
        (2, [2], [1, 0]),                     # [10] vs [01,00]
        (2, [0, 2], [3]),                     # [00,10] vs [11]
        (3, [5, 7], [2]),                     # [101,111] vs [010]
        (4, [15, 7], [0, 8]),                 # [1111,0111] vs [0000,1000]
        (4, [15, 7], [0, 8, 10, 12]),         # [1111,0111] vs [0000,1000,1010,1100]
        (4, [13, 11, 7, 3], [2, 4]),          # [1101,1011,0111,0011] vs [0010,0100]
        (5, [30, 14, 10, 6], [1, 17, 21, 25]) # [11110,01110,01010,00110] vs [00001,10001,10101,11001]
    ]:
        bits0 = [_int_as_bitstring_le(int0, n) for int0 in ints0]
        bits1 = [_int_as_bitstring_le(int1, n) for int1 in ints1]
        amps = [_ints_as_amps(n, ints0), _ints_as_amps(n, ints1)]
        solution_bits = partial(solution, bits0=bits0, bits1=bits1)
        check_distinguish_states(n, 2, [f"{bits0}", f"{bits1}"], amps, solution_bits, 1)


def test_zero_or_wstate(solution=ref.zero_or_wstate if ref_available else None):
    for n in range(1, 5):
        wstate = [0] * 2 ** n
        for ind in range(n):
            wstate[2 ** ind] = 1 / sqrt(n)
        amps = [[1] + [0] * (2 ** n - 1),
                wstate]
        check_distinguish_states(n, 2, ["|0...0⟩", "|W⟩"], amps, solution)


def test_ghz_or_wstate(solution=ref.ghz_or_wstate if ref_available else None):
    for n in range(2, 5):
        ghz = [1] + [0] * (2 ** n - 2) + [1]
        wstate = [0] * 2 ** n
        for ind in range(n):
            wstate[2 ** ind] = 1 / sqrt(n)
        amps = [ghz, wstate]
        check_distinguish_states(n, 2, ["|GHZ⟩", "|W⟩"], amps, solution)


def test_measure_bell_states(solution=ref.measure_bell_states if ref_available else None):
    isq2 = 1 / sqrt(2)
    amps = [[isq2, 0, 0, isq2],
            [isq2, 0, 0, -isq2],
            [0, isq2, isq2, 0],
            [0, -isq2, isq2, 0]
           ]
    check_distinguish_states(2, 4, ["|00⟩ + |11⟩", "|00⟩ - |11⟩", "|01⟩ + |10⟩",  "|01⟩ - |10⟩"], amps, solution)


def test_measure_twoqubit_separable_states(solution=ref.measure_twoqubit_separable_states if ref_available else None):
    amps = [[0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5]
           ]
    check_distinguish_states(2, 4, ["|S0⟩", "|S1⟩", "|S2⟩",  "|S3⟩"], amps, solution)


def test_measure_twoqubit_entangled_states(solution=ref.measure_twoqubit_entangled_states if ref_available else None):
    amps = [[-0.5,  0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5,  0.5],
            [ 0.5,  0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5, -0.5]
           ]
    check_distinguish_states(2, 4, ["|S0⟩", "|S1⟩", "|S2⟩",  "|S3⟩"], amps, solution)


def test_measure_threequbit_states(solution=ref.measure_threequbit_states if ref_available else None):
    omega = exp(2j * pi / 3)
    amps = [[0, 1 / sqrt(3), omega / sqrt(3), 0, omega ** 2 / sqrt(3), 0, 0, 0],
            [0, 1 / sqrt(3), omega ** 2 / sqrt(3), 0, omega / sqrt(3), 0, 0, 0]
           ]
    check_distinguish_states(3, 2, ["|S0⟩", "|S1⟩"], amps, solution)
