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

def circuit_estimate(p):
    qubits = 40
    t = 16
    depth = 1000
    count = 1000

    rng = random.Random(1000)

    measured_qubits = 8
    aArray = np.zeros(measured_qubits, dtype=np.uint8)
    #print("i d r t delta_d delta_t delta_t_prime final_d final_t log_v")

    circ_id = 17
    seed = 42342
    
    for i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, 18, 16, rng)):
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
        #print(d,t,r)
        if i == circ_id:
            #print(d,t,r)
            phases = np.zeros(40, dtype=np.float64)

            total_extent = np.power(4-2*np.sqrt(2), 32)

            phi = np.arccos(np.power(p, 1./16))*2
            #print(p, phi)
            sqrt_remaining_extent_per_t = np.sqrt(4-2*np.sqrt(2))/np.power(np.sqrt(1-np.sin(phi)) + np.sqrt(1-np.cos(phi)), 1/4.)

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
            
            for k in range(32):
                phases[k] = theta
            for k in range(32, 40):
                phases[k] = phi
                
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
            epsTot = 0.05
            deltaTot = 1e-3
            delta_UB = 0.05
            K_UUB = 25
            beta = np.log2(4. - 2.*np.sqrt(2.));
            effectiveT = 32
            t = 40            
            m = np.power(2, beta*effectiveT/2)

            #bestCost = tauStar(p, m, epsTot, deltaTot, t, r)

            #pStarUB, KUB, kUB = estimate2.hackedOptimise(p, m, epsTot, deltaTot, t, r, delta_UB, K_UUB)
            #new_upper_bound = tauStar(p+2*epsTot, m, epsTot, 6*deltaTot/np.power((K_UB+1)*np.pi,2), t, r)
            #with open("optimize_with_phases.txt", "a") as out_file:
            #    print(p, file=out_file)
            pStar, K, k,time = estimate2.optimize_with_phases(epsTot, deltaTot, 40, measured_qubits, r, log_v, m, CH, AG, phases, seed=random.randrange(0,34534), threads=12)
            return pStar, K, k,time
        
def circuit_bounds(p):
    qubits = 40
    t = 16
    depth = 1000
    count = 1000

    rng = random.Random(1000)

    measured_qubits = 8
    aArray = np.zeros(measured_qubits, dtype=np.uint8)
    #print("i d r t delta_d delta_t delta_t_prime final_d final_t log_v")

    circ_id = 17
    seed = 42342
    
    for i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, 18, 16, rng)):
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
        #print(d,t,r)
        if i == circ_id:
            #print(d,t,r)
            phases = np.zeros(40, dtype=np.float64)

            total_extent = np.power(4-2*np.sqrt(2), 32)

            phi = np.arccos(np.power(p, 1./16))*2
            #print(p, phi)
            sqrt_remaining_extent_per_t = np.sqrt(4-2*np.sqrt(2))/np.power(np.sqrt(1-np.sin(phi)) + np.sqrt(1-np.cos(phi)), 1/4.)

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
            
            for k in range(32):
                phases[k] = theta
            for k in range(32, 40):
                phases[k] = phi
                
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
            epsTot = 0.05
            deltaTot = 1e-3
            delta_UB = 0.05
            K_UUB = 25
            beta = np.log2(4. - 2.*np.sqrt(2.));
            effectiveT = 32
            t = 40            
            m = np.power(2, beta*effectiveT/2)

            bestCost = tauStar(p, m, epsTot, deltaTot, t, r)

            pStarUB, KUB, kUB = estimate2.hackedOptimise(p, m, epsTot, deltaTot, t, r, delta_UB, K_UUB)
            #new_upper_bound = tauStar(p+2*epsTot, m, epsTot, 6*deltaTot/np.power((K_UB+1)*np.pi,2), t, r)
            #pStar, K, k = estimate2.optimize_with_phases(epsTot, deltaTot, 40, measured_qubits, r, log_v, m, CH, AG, phases, seed=random.randrange(0,34534), threads=12)
            return p, bestCost, pStarUB, KUB, kUB
            #
            #return upper_cost, K_UB,tau
        
        
            #p = cPSCS.estimate_algorithm_with_arbitrary_phases(magic_samples, equatorial_samples, measured_qubits, log_v, r, seed, CH, AG, phases)
def costFn(s, L, t, r):
    return s*t*t*(t-r) + s*L*r*r*r

def tauStar(p, m, epsTot, deltaTot, t,r):
    epss = np.linspace(epsTot/1000, epsTot, endpoint=False, num = 1000)
    deltas = np.linspace(deltaTot/1000, deltaTot, endpoint=False, num = 1000)
    bestCost = float("inf")
    for eps in epss:
        for delta in deltas:
            s = np.ceil( 2*(np.power(m + np.sqrt(p), 2.)/np.power(np.sqrt(p+eps) - np.sqrt(p), 2.))*np.log(2*np.exp(2)/delta))
            L = np.ceil( np.power((p+eps)/(epsTot-eps), 2.) *np.log(1/(deltaTot-delta)))
            cost = costFn(s,L,t,r)
            if cost < bestCost:
                bestCost = cost
                bestEps = eps
                bestDelta = delta
                bestS = s
                bestL = L
    return bestCost

if __name__ == "__testing__":
    #for prob in np.linspace(0.0,1,100):
    #    bestCost, new_upper_bound, upper_cost, K_UB = circuit(prob)
    #print(prob, bestCost, new_upper_bound, upper_cost, K_UB)
    print("pStar K k pStarUB KUB kUB")
    for prob in [0.015, 0.051, 0.101, 0.157]:
        circuit(prob)
        
    
if __name__ == "__main__":
    epsTot = 0.05
    deltaTot = 0.001
    beta = np.log2(4. - 2.*np.sqrt(2.));
    effectiveT = 32
    t = 40            
    m = np.power(2, beta*effectiveT/2)
    r = 8

    #p, K_UB, upper_cost = estimate2.hackedOptimise(1, m, epsTot, deltaTot, t, r, 0.05, 22)
    #print(p,K_UB, upper_cost)
    #threads = 10
    #prob = 0.4631578947368421
    #prob = 0.46842105263157896
    
    #estimate2.hackedOptimise(prob, m, epsTot, deltaTot, t, r, 0.05, 22)
    #with Pool(12) as p:
    #    probs = np.linspace(0.0,1, 100)
    #    #ans = p.starmap(estimate2.hackedOptimise, [(prob, m, epsTot, deltaTot, t, r, 0.05, 22) for prob in probs])
    #    ans = p.map(circuit_bounds, probs)
    #    for prob, bestCost, pStarUB, KUB, kUB in ans:
    #        print(prob, bestCost, pStarUB, KUB, kUB)
            

    for prob in [0.015, 0.318]:
        pStar, K, k,time = circuit_estimate(prob)
        print(prob, pStar, K, k)
            

    
