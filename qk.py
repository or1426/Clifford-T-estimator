import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit import Aer
import cliffords

import numpy as np

class QiskitSimulator(object):
    """
    Use Qiskit's stabilizer simulator to run one of our circuits
    """
    def __init__(self):
        self.backend = Aer.get_backend("statevector_simulator") #QasmSimulator({"method": "statevector_"})
        ##self.backend_options = {"method": "stabilizer"}

    def run(self, num_qubits, state, composite: cliffords.CompositeCliffordGate):
        circuit = QuantumCircuit(len(state)) 
        #first flip all the bits for which state is 1

        for i, b in enumerate(state):
            if b:
                circuit.x(i)
        projector = None
        for gate in composite.gates:
            d = gate.data()
            if d[0] == "H":
                circuit.h(d[1])
            elif d[0] == "S":
                circuit.s(d[1])
            elif d[0] == "CZ":
                circuit.cz(target_qubit=d[1], control_qubit=d[2])
            elif d[0] == "CX":
                circuit.cx(target_qubit=d[1], control_qubit=d[2])
            elif d[0] == "PZ":                
                projector = d
                break # once we hit a projector the circuit is done                
            else:
                raise TypeError("Only unitary Clifford gates and Z projectors supported! Recieved: {}".format(gate))
        job = qiskit.execute(circuit, backend=self.backend,backend_options={"method": "statevector"})
        statevector = _rearange_state_vector(num_qubits, job.result().get_statevector(circuit))
        if projector:
            #we need to zero everything that we aren't projecting onto
            #this will be half of the entries of the state-vector
            target_qubit = projector[1]
            bits_to_zero = None
            if projector[2] == 0:
                bit_to_zero = '1'
            elif projector[2] == 1:
                bit_to_zero = '0'
            indices_to_zero = [n for n in range(len(statevector)) if np.binary_repr(n,width=num_qubits)[target_qubit] == bit_to_zero]
            #print(target_qubit)
            #print([np.binary_repr(n,width=num_qubits) for n in range(len(statevector))])
            #print(indices_to_zero)
            statevector[indices_to_zero] = 0.
        
        #job = self.backend.run(circuit, backend_options=self.backend_options)
        return statevector


def _rearange_state_vector(num_qubits, statevector):
    #qiskit returns statevectors in a slightly strange order, probably we shuld match qiskit's convention in the future but for now we do not
    a = np.array(range(len(statevector)),dtype=np.int)
    a = [sum([int(bit) * 2**(k) for k, bit in enumerate(np.binary_repr(number,width=num_qubits))]) for number in a]
    return statevector[np.argsort(a)]

def _apply_z0_projector(statevector):
    for i in range(int(len(statevector)/2)):
        statevector[i] = 0
    return statevector
