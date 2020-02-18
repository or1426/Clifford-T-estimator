#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome
from stabstate import StabState
import constants
from cliffords import SGate, CXGate, CZGate, HGate
                    
                
            
def _basis_test():
    print(StabState.basis(N=2))
    print(StabState.basis())
    print(StabState.basis(N=5))

    print(StabState.basis(s=[0,1,0]))

    print(StabState.basis(s=[0,1,0,1,0]))

    print(StabState.basis(N=2,s=[0,1,0,1,0]))
    print(StabState.basis(N=5,s=[0,1,0,1,0]))
    print(StabState.basis(N=10, s=[0,1,0,1,0]))



if __name__ == "__main__":
    s0 = MeasurementOutcome(np.array([0], dtype=np.uint8))
    s1 = MeasurementOutcome(np.array([1], dtype=np.uint8))
    
    s00 = MeasurementOutcome(np.array([0,0], dtype=np.uint8))
    s01 = MeasurementOutcome(np.array([0,1], dtype=np.uint8))
    s10 = MeasurementOutcome(np.array([1,0], dtype=np.uint8))
    s11 = MeasurementOutcome(np.array([1,1], dtype=np.uint8))

    sgate = SGate(0)
    cnot = CXGate(1,0)
    
    state = StabState.basis(s=[1,1])
    #state = cnot.apply(state)
    #state = SGate(1).apply(sgate.apply(state))
    #state = CXGate(control=0,target=1).apply(HGate(0).apply(state))

    state = CZGate(0, 1).apply(state)
    
    #print(s01.overlap(state))
    #print(s.overlap(state))
    
    #state = StabState.basis(s=[1,1])
    #print(state.g)
    #state = SGate(1).apply(state)
    #print(state.g)
    #state = cnot.apply(state)

    
    
    print(s00.overlap(state))
    print(s01.overlap(state))
    print(s10.overlap(state))
    print(s11.overlap(state))
    
