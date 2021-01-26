#! /usr/bin/env python3

import util
import cPSCS

from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate

import time
import random

import numpy as np
from multiprocessing import Pool
import sycamore

import sys

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
        
    return cPSCS.compress_algorithm_no_state_output(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray)
    
if __name__ == "__main__":

    cycle_count = int(sys.argv[1])
    
    qubits = 53
    measured_qubits = 3

    #depth d r t delta_d delta_t delta_t_prime v time
    num = 200
        
    filename = "run_data_2/data_set_12-{}-{}.txt".format(cycle_count, sys.argv[2])
    
    #cycles = np.array([1,2,3,4,6,8,10,12, 16, 20, 80],dtype=np.int)

    #d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, pyChState, pyAGState, _ = out
    
    rng = random.Random(time.monotonic())
    
    with open(filename, "a") as f:
        print("cycle_count d r t delta_d delta_t delta_t_prime final_d final_t v time",file=f,flush=True)
        
        for i in range(num):
            circ,_,_ = sycamore.sycamore_circuit(6, 9, cycle_count, rng=rng)

            out = compress_wrapper(qubits, measured_qubits, circ)
            d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v,time =  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
            if type(out) == tuple:                    
                d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v = out            
            print(cycle_count, d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, time,file=f,flush=True)
