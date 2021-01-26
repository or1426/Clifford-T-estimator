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
import estimate2
from multiprocessing import Pool
from estimate2 import LMin,eps2


import time


def circuit(p):
    qubits = 50
    base_U_t = 26
    depth = 1000
    count = 880

    rng = random.Random(1000)

    measured_qubits = 8
    aArray = np.zeros(measured_qubits, dtype=np.uint8)
    #print("i d r t delta_d delta_t delta_t_prime final_d final_t log_v")

    circ_id = 878
    
    
    for i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, count, base_U_t, rng)):
        #print(t)
        uDagger  = circ.inverse()
        fullCirc = CompositeCliffordGate()
        fullCirc.gates = circ.gates + uDagger.gates

        for j in range(measured_qubits):
           fullCirc.gates.append(HGate(j))
           fullCirc.gates.append(TGate(j))
           fullCirc.gates.append(HGate(j))
        #[print(g) for g in fullCirc.gates]
        gateArray = np.zeros(len(fullCirc.gates), dtype=np.uint8)
        controlArray = np.zeros(len(fullCirc.gates), dtype=np.uint)
        targetArray = np.zeros(len(fullCirc.gates), dtype=np.uint)
        
        for j, gate in enumerate(fullCirc.gates):
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

        d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, CH, AG = cPSCS.compress_algorithm_no_region_c_constraints(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        #if r < 11:
        #    print(i, d,t,r)
        if i == circ_id:
            print(i, d, t, r)
            
            phases = np.zeros(60, dtype=np.float64)

            total_extent = np.power(4-2*np.sqrt(2), 52)

            phi = np.arccos(np.power(p, 1./16))*2
            #print(p, phi)
            sqrt_remaining_extent_per_t = np.sqrt(4-2*np.sqrt(2))/np.power(np.sqrt(1-np.sin(phi)) + np.sqrt(1-np.cos(phi)), 8./52.)
            
            #so now we need the theta for which sqrt(1-sin(theta)) + sqrt(1-cos(theta)) = remaining_extent_per_t
            # this function is monotone so this is easy
            
            precision = 1e-15
            lower_bound = 0
            upper_bound = np.pi/4

            width = upper_bound - lower_bound
            midpoint = lower_bound + width/2.
            while width > precision:
                midpoint = lower_bound + width/2.
                if np.sqrt(1-np.sin(midpoint)) + np.sqrt(1-np.cos(midpoint)) > sqrt_remaining_extent_per_t:
                    upper_bound = midpoint
                else:
                    lower_bound = midpoint
                width = upper_bound - lower_bound
                
            theta = midpoint
            #print(phi, theta, np.pi/4)
            #print( (np.sqrt(1-np.sin(phi)) + np.sqrt(1-np.cos(phi)))**16, (np.sqrt(1-np.sin(theta)) + np.sqrt(1-np.cos(theta)))**64,  ((np.sqrt(1-np.sin(phi)) + np.sqrt(1-np.cos(phi)))**16)*((np.sqrt(1-np.sin(theta)) + np.sqrt(1-np.cos(theta)))**64),total_extent)
            total_extent = 1.
            for k in range(52):
                phases[k] = theta
            for k in range(52, 60):
                phases[k] = phi

            for k in range(60):
                total_extent *= ((np.sqrt(1-np.sin(phases[k])) + np.sqrt(1-np.cos(phases[k])))**2)
            correct_extent = (4-2*np.sqrt(2))**52
            print("These numbers should be the same:", total_extent, correct_extent)
            #for k in range(measured_qubits):
            #    fullCirc.gates.append(PauliZProjector(target=k,a=0))
                
            #sim = qk.QiskitSimulator()
            #qkvec = sim.run(15, np.zeros(15,dtype=np.uint8), fullCirc, phases=phases)
            #print(qkvec.conjugate() @ qkvec)
            #ext_sqrt = 1.
            #for phase in phases:
            #    ext_sqrt *= (np.sqrt(1-np.sin(phase)) +  np.sqrt(1-np.cos(phase)))
            #print(ext_sqrt)
            #print(phases)
            epsTot = 0.02
            deltaTot = 1e-3
            beta = np.log2(4. - 2.*np.sqrt(2.));
            effectiveT = 52
            t = 60
            m = np.power(2, beta*effectiveT/2)
            
            optimize_prob, optimize_cost = estimate2.optimize_with_phases(epsTot, deltaTot, 40, measured_qubits, r, log_v, m, CH, AG, phases, seed=random.randrange(0,34534), threads=12)
        
        

def costFn(s, L, t, r):
    return s*t*t*(t-r) + s*L*r*r*r

if __name__ == "__main__":
            
    p = 0.05
    print("-----------------")
    print(p)
    print("-----------------")
    circuit(p)

    

