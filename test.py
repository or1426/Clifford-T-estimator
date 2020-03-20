#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome, PauliZProjector
from stabstate import StabState
import constants
from cliffords import SGate, CXGate, CZGate, HGate, CompositeGate, SwapGate
import itertools                    
import util
import random
import qk

def eq(a,b,e=1e-10):
    return linalg.norm(reconstructed_vector1 - reconstructed_vector2) < 10e-10
    

from numpy import linalg

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
    qubits, depth, N = 5, 100, 1000
    np.set_printoptions(linewidth=100)
    sketch = 0
    zero = 0
    equal12 = 0
    n12 = 0
    equal23 = 0 
    n23 = 0
    equal31 = 0
    n31 = 0
    for i, (vector, circuit) in enumerate(zip(random.choices(list(itertools.product(range(2), repeat=qubits)), k=N),
                               util.random_clifford_circuits(qubits=qubits, depth=depth, N=N))):
        
        #print(i)
        #a, b = random.sample(range(qubits), 2)
        #projector = PauliZProjector(random.choice(range(qubits)), random.choice([0,1]))]
        projector = PauliZProjector(0,0)
        #print(projector)
        state = StabState.basis(s=vector) | circuit | projector

        reconstructed_vector1 = np.zeros(2**qubits, dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
           reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)

        
        if state.phase > 10e-10: #we don't care if the state is just 0
            if not ((state.s == 0) & (state.v == 0)).any() and not ((~state.v%2)  * state.s).sum() > 1:
                print(state.tab())
                sketch += 1
            else :
                if ((~state.v % 2) * state.s).sum() > 1: 
                    # we have two qubits with v == 0 and s == 1
                    #we will apply a cnot between them, giving us a qubit with v == 0 and s == 0
                    #and cancel out this cnot by putting exactly the same one in UC
                    p, q = np.flatnonzero((~state.v % 2) * state.s)[:2]
                    CXGate(p,q).rightMultiplyC(state)
                    state.s[p] = 0
                if ((state.s == 0) & (state.v == 0)).any():    
                    p = np.flatnonzero((state.s == 0) & (state.v == 0))[0]
                    if p != projector.target:
                        swap = CompositeGate([CXGate(p, projector.target), CXGate(projector.target, p), CXGate(p, projector.target)])
                        permutation = list(range(qubits))
                        permutation[p] = projector.target
                        permutation[projector.target] = p
                        
                        state.v = state.v[permutation]
                        state.s = state.s[permutation]
        
                        for gate in swap.gates:
                            gate.rightMultiplyC(state)
                            
            reconstructed_vector2 = np.zeros(2**qubits, dtype=np.complex)
            for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
                reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
            state = state.delete_qubit(projector.target)
            reconstructed_vector3 = np.zeros(2**(qubits), dtype=np.complex)
            for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
               if t[projector.target] == projector.a:
                   t2 = t[:projector.target] + t[projector.target+1:]
                   reconstructed_vector3[i] = MeasurementOutcome(np.array(t2, dtype=np.uint8)).apply(state)

            if eq(reconstructed_vector1, reconstructed_vector2):
                equal12 += 1
            else:
                n12 +=1
            if eq(reconstructed_vector2, reconstructed_vector3):
                equal23 += 1
            else:
                n23 +=1
            if eq(reconstructed_vector3, reconstructed_vector1):
                equal31 += 1
            else:
                n31 +=1            
        
        else:
            zero += 1

    print("zeros: ", zero)
    print("sketch: ", sketch)
    print("1 == 2:", equal12, n12)
    print("2 == 2:", equal23, n23)
    print("3 == 1:", equal31, n31)
