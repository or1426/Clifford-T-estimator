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
            epsTot = 0.06
            deltaTot = 1e-3
            beta = np.log2(4. - 2.*np.sqrt(2.));
            effectiveT = 32
            t = 40            
            m = np.power(2, beta*effectiveT/2)

            epss = np.linspace(epsTot/1000, epsTot, endpoint=False, num = 1000)
            deltas = np.linspace(deltaTot/1000, deltaTot, endpoint=False, num = 1000)
            bestCost = float("inf")
            bestS = None
            bestL = None
            bestEps = None
            bestDelta = None

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

            optimize_prob, optimize_cost = estimate2.optimize_with_phases(epsTot, deltaTot, 40, measured_qubits, r, log_v, m, CH, AG, phases, seed=1463, threads=10)
            return optimize_prob, optimize_cost, bestCost
        
        
            #p = cPSCS.estimate_algorithm_with_arbitrary_phases(magic_samples, equatorial_samples, measured_qubits, log_v, r, seed, CH, AG, phases)
def costFn(s, L, t, r):
    return s*t*t*(t-r) + s*L*r*r*r

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
    # with Pool(threads) as p:
    #     probs = np.linspace(0.0,1, 100)
    #     ans = p.starmap(estimate2.hackedOptimise, [(prob, m, epsTot, deltaTot, t, r, 0.05, 22) for prob in probs])
    #     ans2 = p.map(circuit, probs)
    #     for (prob, k, upper_cost), lower_cost in zip(ans, ans2):
    #         print(prob, lower_cost, upper_cost, k)
            
    for p in [0.015, 0.051, 0.101, 0.157, 0.227,0.318]: # 0.4,0.3,0.2]:
        print("-----------------")
        print(p)
        print("-----------------")
        optimize_prob, optimize_cost, lower_cost = circuit(p)
        #upper_cost = estimate2.hackedOptimise(p, m, epsTot, deltaTot, t, r, 0.05,22)[1]
        print(p, lower_cost, optimize_prob, optimize_cost)
        #     print(p, lower_cost, upper_cost)



    # epsTot = 0.05
    # deltaTot = 1e-3
    # beta = np.log2(4. - 2.*np.sqrt(2.));
    # effectiveT = 32
    # t = 40            
    # m = np.power(2, beta*effectiveT/2)
    
    # bestEps = float("inf")
    
    # bestS = None
    # bestL = None
    # etas = np.linspace(1/1000, 1, endpoint=False, num = 100)
    # #9389 87562
    # tau = 9389 * t *t * (t-r) + 87562*9389*r*r*r
    # print("tau = ", tau)
    # deltaTarg = 0.000607927101854026
    
    # for eta in etas:
    #     s_min = 1
    #     s_max = int(np.floor(tau / ( t*t*(t-r) + r*r*r*np.ceil(LMin(deltaTarg, eta)))))

    #     for s in np.linspace(s_min, s_max, endpoint=False, num = 100):
    #         L = (tau - s*t*t*(t-r))/(s*r*r*r)
    #         eps = eps2(1, deltaTarg, eta, s, tau, m, t, r, 1e-15)
    #         if eps < bestEps:
    #             bestEps = eps
    #             bestS = s
    #             bestL = L
                
    # print(822117376.0, costFn(bestS, bestL, t,r))
    # print(bestS, bestL, bestEps, eps2(1, deltaTarg, 0.576622009277343, 87562, tau, m, t, r, 1e-15))
            
            

    
