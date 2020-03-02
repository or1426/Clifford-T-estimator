#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome, PauliZProjector
from stabstate import StabState
import constants
from cliffords import SGate, CXGate, CZGate, HGate, CompositeGate
import itertools                    
import util
import random
import qk

if __name__ == "__main__":

    #state = StabState.basis(s=[0,0]) # generate a state |00>
    #state = state | HGate(0) | CXGate(target=1, control=0) #pass it through a Hadamard and a CNOT

    #state.__str__ method prints N F G M gamma v s omega
    #N is a number (number of qubits), F G and M are block matrices, gamma, v and s are column vectors and omega is a complex number
    #this is all the information required to describe the state
    #print(state)

    
    #for t in itertools.product([0,1], repeat=2):
    #   print(t, state | MeasurementOutcome(np.array(t, dtype=np.uint8))) # print out the overlaps with <00|, <01|, <10| and <11

    #generate some random computational basis states, some random Clifford circuits and apply them 
    qubits, depth, N = 10, 20, 1000
    np.set_printoptions(linewidth=100)
    #(1, 1)
    #[CX(1, 0), H(0), H(1)]
    
    #gates = [CXGate(target=0,control=1), SGate(1), CZGate(target=1, control=0), CXGate(target=1,control=0)]
    

    # gates = [HGate(1), SGate(1), HGate(0), CZGate(0,1), HGate(1)]
    # print(CompositeGate(gates))
    # vector = np.array([1,1], dtype=np.uint8)
    # state = StabState.basis(s=vector)
    # reconstructed_vector = np.zeros(2**qubits, dtype=np.complex)
    # for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
    #     reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
    # tol = 1e-14
    # reconstructed_vector.real[abs(reconstructed_vector.real) < tol] = 0.0
    # reconstructed_vector.imag[abs(reconstructed_vector.imag) < tol] = 0.0
    # print(reconstructed_vector)
    
    # for gate in gates:
    #     print(state)
    #     print("Applying {}".format(gate))
    #     gate.apply(state)
    #     reconstructed_vector = np.zeros(2**qubits, dtype=np.complex)
    #     for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
    #         print(t, MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state))
    #         reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
    #     tol = 1e-14
    #     reconstructed_vector.real[abs(reconstructed_vector.real) < tol] = 0.0
    #     reconstructed_vector.imag[abs(reconstructed_vector.imag) < tol] = 0.0
    #     print(reconstructed_vector)
    # print(state)
    # sim = qk.QiskitSimulator()
    # qk_vector = sim.run(qubits, vector, CompositeGate(gates))
    # qk_vector.real[abs(qk_vector.real) < tol] = 0.0
    # qk_vector.imag[abs(qk_vector.imag) < tol] = 0.0

    # print(qk_vector) 
    
    
    for vector, circuit in zip( random.choices(list(itertools.product(range(2), repeat=qubits)), k=N),
                                util.random_clifford_circuits(qubits=qubits, depth=depth, N=N)):
        state = StabState.basis(s=vector)
        
        sim = qk.QiskitSimulator()
        state | circuit #project onto <0| on the zeroth qubit (apply 1/2 (I + (-1)**0 Z_0))
        
        reconstructed_vector = np.zeros(2**qubits, dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
            #i = int(sum([digit*2**k  for k,digit in enumerate(t)]))
            reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
        sim = qk.QiskitSimulator()

        tol = 1e-14
        reconstructed_vector.real[abs(reconstructed_vector.real) < tol] = 0.0
        reconstructed_vector.imag[abs(reconstructed_vector.imag) < tol] = 0.0

        qk_vector = sim.run(qubits, vector, circuit)
        qk_vector.real[abs(qk_vector.real) < tol] = 0.0
        qk_vector.imag[abs(qk_vector.imag) < tol] = 0.0

        if (reconstructed_vector - qk_vector).conjugate() @ (reconstructed_vector - qk_vector) > 10e-10:
            print(vector)
            print(circuit)
            print(reconstructed_vector)
            print(qk_vector)
            print(abs(reconstructed_vector - qk_vector))
            break
        
        
        #print(state.F[0] * state.v *state.phase)

    #sim = qk.QiskitSimulator()

    #job = sim.run([0,0], HGate(0) | CXGate(control=0,target=1))
    #print(job)    


    
