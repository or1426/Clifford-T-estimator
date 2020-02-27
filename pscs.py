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

if __name__ == "__main__":

    state = StabState.basis(s=[0,0]) # generate a state |00>
    state = state | HGate(0) | CXGate(target=1, control=0) #pass it through a Hadamard and a CNOT

    #state.__str__ method prints N F G M gamma v s omega
    #N is a number (number of qubits), F G and M are block matrices, gamma, v and s are column vectors and omega is a complex number
    #this is all the information required to describe the state
    print(state)

    
    for t in itertools.product([0,1], repeat=2):
        print(t, state | MeasurementOutcome(np.array(t, dtype=np.uint8))) # print out the overlaps with <00|, <01|, <10| and <11

    #generate some random computational basis states, some random Clifford circuits and apply them 
    qubits, depth, N = 3, 5, 1

    for state, circuit in zip( [StabState.basis(s=s) for s in random.choices(list(itertools.product(range(2), repeat=qubits)), k=N)],
                               util.random_clifford_circuits(qubits=qubits, depth=depth, N=N)):
        s = state | circuit | PauliZProjector(0,0) #project onto <0| on the zeroth qubit (apply 1/2 (I + (-1)**0 Z_0))
        print(state)
        
    


    
