#! /usr/bin/env python3

import util
import cPSCS

from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate

import time

import numpy as np

def tests_wrapper(qubits, samples, circ):
    depth = len(circ.gates)
    gateArray = np.zeros(depth, dtype=np.uint8)
    controlArray = np.zeros(depth, dtype=np.uint)
    targetArray = np.zeros(depth, dtype=np.uint)
    
    for j, gate in enumerate(circ.gates):
        if isinstance(gate, CXGate):
            gateArray[j] = 88 #X
            controlArray[j] = gate.control
            targetArray[j] = gate.target
        elif isinstance(gate, CZGate):
            gateArray[j] = 90 #Z
            controlArray[j] = gate.control
            targetArray[j] = gate.target
        elif isinstance(gate, SGate):
            gateArray[j] = 115 #s
            targetArray[j] = gate.target
        elif isinstance(gate, HGate):
            gateArray[j] = 104 #h
            targetArray[j] = gate.target
        elif isinstance(gate, TGate):
            gateArray[j] = 116 # t
            targetArray[j] = gate.target
        
    return cPSCS.equatorial_inner_product_tests(qubits, samples, gateArray, controlArray, targetArray)
    

if __name__ == "__main__":
    qubits = 50
    eq_samples = 100
    ch_samples = 10
    clifford_depth = 10000
    

    
    for circ in util.random_clifford_circuits(qubits, clifford_depth, ch_samples):

        new_time, old_time, _ = tests_wrapper(qubits, eq_samples, circ)

            
        with open("equatorial_timings.txt", "a") as f:
            print(qubits, new_time, old_time,file=f)
                
    
