#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from chstate import CHState
import constants
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate
import itertools                    
import util
import random
import pickle
import math
import qk
import sys
from matplotlib import pyplot as plt
import cPSCS
import estimate
from multiprocessing import Pool


import time

def basisTuple(n, *args):
    v = [0]*n
    for arg in args:
        v[arg] = 1
    return tuple(v)

def eq(a,b,eps=1e-10):
    return linalg.norm(a - b) < eps
    

from numpy import linalg

if __name__ == "__main__":
    # do some simulations make some graphs
    qubits = 20
    measured_qubits = 5
    print("n = ", qubits)
    print("w = ", measured_qubits)
    circs = 1
    depth = 1000
    t = 10
    print("t =", t)

    seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(measured_qubits)] + [0 for _ in range(qubits - measured_qubits)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    #epss = [0.1, 0.01,0.001]
    epss = [.01]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    #random.seed(15111)
    #seed = random.randint(100,100000)
    seed = 1923
    #seed = 21517
    print("seed=",seed)
    random.seed(seed)
    threads = 10
    
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        gateArray = np.zeros(depth, dtype=np.uint8)
        controlArray = np.zeros(depth, dtype=np.uint)
        targetArray = np.zeros(depth, dtype=np.uint)
        actual_t_count = 0
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
                actual_t_count += 1
        for i, (a,m) in enumerate(zip(aArray, mask)):
           if m:
               circ | PauliZProjector(target=i, a=a)

        phases = np.zeros(actual_t_count, dtype=np.float) + np.pi/4.
        print(phases)
        start = time.monotonic_ns()
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = (qk_vector.conjugate() @ qk_vector).real
        qk_time = time.monotonic_ns() - start
        print(qk_val, qk_time)

        d_time = time.monotonic()
        comp_val = cPSCS.calculate_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))        
        comp_time = time.monotonic() - d_time
        print("compute_algorithm: {:.16f}".format(comp_val), "took", comp_time, "seconds")
        
        d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, CH, AG, magic_arr = cPSCS.compress_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        for eps in epss:
            print(eps)
            meps = 0.5*eps
            magic_samples = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + meps) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
            equatorial_samples = int(round((((qk_val + meps)/(eps-meps))**2)*(-np.log(delta))))
            best_ratio = 0.5            
            for ratio in range(1,99):
                meps2 = ratio*eps/100.
                magic_samplesr = 2*math.pow((math.pow(2.,gamma*t/2) + math.sqrt(qk_val))/(math.sqrt(qk_val + meps2) - math.sqrt(qk_val)), 2.)*math.log(2*math.e*math.e/(delta/2))
                equatorial_samplesr = math.pow((qk_val + meps2)/(eps-meps2), 2.)*math.log(1/(delta/2.))
                magic_samplesr = int(round(magic_samplesr))
                equatorial_samplesr = int(round(equatorial_samplesr))

                if 0 < magic_samplesr*equatorial_samplesr < magic_samples*equatorial_samples:
                    best_ratio = ratio
                    magic_samples = magic_samplesr
                    equatorial_samples = equatorial_samplesr
            
            print(magic_samples, equatorial_samples, best_ratio)
                        
            p1 = cPSCS.estimate_algorithm(magic_samples, equatorial_samples, measured_qubits, log_v, r, seed, CH, AG)

            p2 = cPSCS.estimate_algorithm_with_arbitrary_phases(magic_samples, equatorial_samples, measured_qubits, log_v, r, seed, CH, AG, phases)

            print(eps, p1, p2)                
