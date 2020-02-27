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
    qubits, depth, N = 3, 5, 1
    q = 0
    for state, circuit in zip( [StabState.basis(s=s) for s in random.choices(list(itertools.product(range(2), repeat=qubits)), k=N)],
                               util.random_clifford_circuits(qubits=qubits, depth=depth, N=N)):
        s = state | circuit | PauliZProjector(q,0)
        print(state)
        
    
    state = StabState.basis(s=[0,0])
    state = state | HGate(0) | CXGate(target=1, control=0)

    for t in itertools.product([0,1], repeat=2):
        print(t, state | MeasurementOutcome(np.array(t, dtype=np.uint8)))
    #print(state1 | s01 )
    #print(state1 | s10 )
    #print(state1 | s11 )
       
    #print(StabState.basis(s=[0,1]) | HGate(0) | CXGate(control=0, target=1) | MeasurementOutcome(np.array([1,1], dtype=np.uint8))) 


    
