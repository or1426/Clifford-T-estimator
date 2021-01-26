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
    

if __name__ == "__multi__":
    qubits = 53
    t = 60
    measured_qubits = 3

    #depth d r t delta_d delta_t delta_t_prime v time
    num = 1000
        
    filename = "run_data/data_set_9.txt"
    clifford_depths = np.array([1,2,3,5,7,10,20,50,200,1000],dtype=np.int)
    threads = 4

    #d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, pyChState, pyAGState, _ = out
    
    with open(filename, "a") as f:
        print("depth d r t delta_d delta_t delta_t_prime final_d final_t log_v time",file=f)
    for depth in clifford_depths:
        print(depth)
        circs = list(util.random_clifford_circuits_with_fixed_T_positions(qubits, depth, num, t))

        output = None
        with Pool(threads) as p:                
            output = p.starmap(compress_wrapper, [(qubits, measured_qubits, c) for c in circs])

        with open(filename, "a") as f:
            for out in output:        
                d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v,time =  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
                if type(out) == tuple:                    
                    d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v = out            
                print(depth, d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, time,file=f)
                
    

if __name__ == "__main__":
    qubits = 53
    t = 60
    measured_qubits = 3

    #depth d r t delta_d delta_t delta_t_prime v time
    num = 1000
        
    filename = "run_data/data_set_9.txt"
    clifford_depths = np.array([1,2,3,5,7,10,20,50,200,1000],dtype=np.int)


    #d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, pyChState, pyAGState, _ = out
    
    with open(filename, "a") as f:
        print("depth d r t delta_d delta_t delta_t_prime final_d final_t log_v time",file=f)
        for depth in clifford_depths:
            print(depth)
            for circ in util.random_clifford_circuits_with_fixed_T_positions(qubits, depth, num, t):
                out = compress_wrapper(qubits, measured_qubits, circ)
                d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v,time =  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
                if type(out) == tuple:                    
                    d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v = out            
                print(depth, d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, time,file=f)
                
    

