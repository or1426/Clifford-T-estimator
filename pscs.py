#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome
from stabstate import StabState
import constants
from cliffords import SGate, CXGate, CZGate, HGate
                    
                
if __name__ == "__main__":
    s00 = MeasurementOutcome(np.array([0,0], dtype=np.uint8))
    
    s01 = MeasurementOutcome(np.array([0,1], dtype=np.uint8))
    s10 = MeasurementOutcome(np.array([1,0], dtype=np.uint8))
    s11 = MeasurementOutcome(np.array([1,1], dtype=np.uint8))

    sgate = SGate(0)

    state1 = StabState.basis(s=[0,1])
    state2 = StabState.basis(s=[0,1])

    state1 = state1 | HGate(0) | CXGate(target=1, control=0)

    
    #state2 = state2 | entangler 
    
    #python expressions are evaluated left to right so this is (state | HGate(0)) | CXGate(0,1)

    
    #print(s01.overlap(state))
    #print(s.overlap(state))
    
    #state = StabState.basis(s=[1,1])
    #print(state.g)
    #state = SGate(1).apply(state)
    #print(state.g)
    #state = cnot.apply(state)
    
    print(state1 | s00 )
    print(state1 | s01 )
    print(state1 | s10 )
    print(state1 | s11 )
       
    #print(StabState.basis(s=[0,1]) | HGate(0) | CXGate(control=0, target=1) | MeasurementOutcome(np.array([1,1], dtype=np.uint8))) 


    
