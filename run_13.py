#! /usr/bin/env python3

import util
import cPSCS

from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate

import time

import numpy as np
from multiprocessing import Pool

def compress_wrapper(qubits, measured_qubits, circ):
    depth = len(circ.gates)
    gateArray = np.zeros(depth, dtype=np.uint8)
    controlArray = np.zeros(depth, dtype=np.uint)
    targetArray = np.zeros(depth, dtype=np.uint)
    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    
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
        
    return cPSCS.compress_algorithm(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray)                        

if __name__ == "__main__":
    qubits = 55
    t = 80
    measured_qubits = 5

    #depth d r t delta_d delta_t delta_t_prime v time
    num = 900
        
    filename = "run_data/data_set_13.txt"
    clifford_depth = 10000 


    #d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, pyChState, pyAGState, _ = out
    
    with open(filename, "a") as f:
        print("depth d r t delta_d delta_t delta_t_prime final_d final_t v time",file=f)
        for i, circ in enumerate(util.random_clifford_circuits_with_fixed_T_positions(qubits, clifford_depth, num, t)):
            print(i)
            out = compress_wrapper(qubits, measured_qubits, circ)
            d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v,time =  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
            if type(out) == tuple:                    
                d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v = out            
            print(clifford_depth, d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, time,file=f)
                
