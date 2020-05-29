#! /usr/bin/env python3

from __future__ import annotations

import numpy as np

from measurement import MeasurementOutcome, PauliZProjector
from chstate import CHState
import constants
from cliffords import SGate, CXGate, CZGate, HGate, CompositeGate, SwapGate
import itertools                    
import util
import random
import time
from copy import deepcopy
import factoring

from numpy import linalg

def eq(a,b,eps=1e-10):
    """
    Slightly dodgy method of testing if two vectors are equal
    we say they are close enough to equal of the norm of their difference is less than eps
    """
    return linalg.norm(a-b) < eps
    

if __name__ == "__main__":
    #how many qubits, clifford gates and overall iterations in our test

    depth, N = 100, 100
    qubits = np.array(range(5,10,1), dtype=np.int)
    
    #apply_time = np.zeros_like(qubits, dtype=np.float)
    #project_time = np.zeros_like(qubits, dtype=np.float)
    #factor_time = np.zeros_like(qubits, dtype=np.float)
    #factor_time2 = np.zeros_like(qubits, dtype=np.float)
    nonzeros = np.zeros_like(qubits, dtype=np.int)
    bad = np.zeros_like(qubits, dtype=np.int)
    good = np.zeros_like(qubits, dtype=np.int)
    
    for count, q in enumerate(qubits):
        print(q)
        for vector, circuit in zip(np.random.randint(2, size=(N,q)), util.random_clifford_circuits(qubits=q, depth=depth, N=N)):
            projector = PauliZProjector(0,0)

            #form a state by passing our initial state through the random Clifford circuit and then apply the projector
            #t0 = time.perf_counter()
            state = CHState.basis(s=vector) | circuit
            #t1 = time.perf_counter()
            state = state | projector        
            #t2 = time.perf_counter()
            #we don't do anything else if it turned out our state was orthogonal to the support of the projector
            if abs(state.phase) > 10e-10:                    
                state2 = deepcopy(state)
                #t3 = time.perf_counter()
                
                nonzeros[count] += 1
                factoring.gaussian_eliminate(state)
                #t4 = time.perf_counter()

                factoring.reduce_column(state2,0)

                reconstructed_vector1 = np.zeros(2**q, dtype=np.complex)
                for i, t in enumerate(itertools.product([0,1], repeat=q)):
                    reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)

                reconstructed_vector2 = np.zeros(2**q, dtype=np.complex)
                for i, t in enumerate(itertools.product([0,1], repeat=q)):
                    reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state2)

                if not eq(reconstructed_vector1, reconstructed_vector2):
                    bad[count] += 1
                else:
                    good[count] += 1 
                #t5 = time.perf_counter()
               
                
                #apply_time[i] += t1-t0
                #project_time[i] += t2-t1
                #factor_time[i] += t4-t3
                #factor_time2[i] += t5-t4
    print(nonzeros)
    print(good)
    print(bad)
    # from matplotlib import pyplot as plt

    # plt.plot(qubits, apply_time/nonzeros, label="apply time",color="r")
    # plt.plot(qubits, project_time/nonzeros, label="project time",color="b")
    # plt.plot(qubits, factor_time/nonzeros, label="factor time",color="g")
    # plt.plot(qubits, factor_time2/nonzeros, "--", label="factor time 2", color="g")

    # plt.yscale("log")
    # plt.legend()
    # plt.grid()
    # plt.show()
