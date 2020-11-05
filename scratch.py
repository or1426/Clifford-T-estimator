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
#import qk

import cPSCS

from matplotlib import pyplot as plt
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
            
if __name__ == "__multi__":
    qubits = 16
    print(qubits)
    circs = 1
    depth = 5000
    t= 12
    print("t=", t)
    #magic_samples = 1000
    #equatorial_samples = 1000
    #seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(8)] + [0 for _ in range(8)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    epss = [0.1, 0.01, 0.001]
    #epss = [0.0001]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    random.seed(121)
    threads = 10
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
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

        for i, (a,m) in enumerate(zip(aArray, mask)):
            if m:
                circ | PauliZProjector(target=i, a=a)
        d_time = time.monotonic_ns()
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        qk_time += time.monotonic_ns() - d_time
        print(qk_val)
        
        for eps in epss:
            print(eps)
            meps = 0.5*eps
            magic_samples = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + meps) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
            equatorial_samples = int(round((((qk_val + meps)/(eps-meps))**2)*(-np.log(delta))))
            for ratio in range(1,9):
                meps2 = ratio*eps/10.
                magic_samplesr = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + meps2) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
                equatorial_samplesr = int(round((((qk_val + meps2)/(eps-meps2))**2)*(-np.log(delta))))

                if 0 < magic_samplesr*equatorial_samplesr < magic_samples*equatorial_samples:
                    magic_samples = magic_samplesr
                    equatorial_samples = equatorial_samplesr

                    
            def run_algorithm_1(seed):
                d_time = time.monotonic()
                v1 = cPSCS.magic_sample_1(qubits,magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
                s1_time = time.monotonic() - d_time

                return (v1, s1_time)

            print(magic_samples)
            print(equatorial_samples)

            with Pool(threads) as p:
                ans = p.map(run_algorithm_1, [random.randrange(1,10000) for _ in range(threads)])
                total_time = 0
                mean_val = 0
                
                for val, time_val in ans:
                    mean_val += val/threads
                    total_time += time_val
                    
                print(ans)
                print(mean_val, total_time)
                print("-----------------------------------")
        
if __name__ == "__main__":
    # do some simulations make some graphs
    qubits = 4
    print(qubits)
    circs = 1
    depth = 5000
    t= 2
    print("t=", t)
    #magic_samples = 1000
    #equatorial_samples = 1000
    seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(2)] + [0 for _ in range(2)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    #epss = [0.1, 0.01,0.001]
    epss = [0.1]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    random.seed(14321)
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
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

        for i, (a,m) in enumerate(zip(aArray, mask)):
            if m:
                circ | PauliZProjector(target=i, a=a)
        #d_time = time.monotonic_ns()
        #sim = qk.QiskitSimulator()
        #qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        #qk_val = qk_vector.conjugate() @ qk_vector
        #qk_time += time.monotonic_ns() - d_time
        #print(qk_val)
        qk_val = 0.003
        
        #magic_samples = int(round(((2*(math.pow(2,gamma*t/2)+qk_val)**2)/((math.sqrt( (1+relative_eps)*qk_val) - math.sqrt(qk_val))**2))/(math.log(delta/(2*relative_eps*relative_eps*qk_val*qk_val)))))
        #equatorial_samples = int(round(((((1+relative_eps))/(relative_eps))**2)*(-np.log(delta))))
        for eps in epss:
            print(eps)
            meps = 0.9*eps
            magic_samples = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + eps) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
            equatorial_samples = int(round((((qk_val + meps)/(eps-meps))**2)*(-np.log(delta))))
            print(magic_samples)
            print(equatorial_samples)
            #print(qk_val)
            #d_time = time.monotonic_ns()
            
            #v1 = cPSCS.magic_sample_1(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            #s1_time += time.monotonic_ns() - d_time
            #print("v1:", v1)
            #print(s1_time/1e9)
            
            #d_time = time.monotonic_ns()
            #v2 = cPSCS.magic_sample_2(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            #s2_time += time.monotonic_ns() - d_time
            
            #print("v2:", v2)

            v3 = cPSCS.main_simulation_algorithm(qubits, magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), 8, np.copy(aArray))

            print("v3: ", v3)

            
            # print(s2_time/1e9)
            
            #print()
            #print(qk_time/1e9, s1_time/1e9,s2_time/1e9)
            #print("-------------------")

