#! /usr/bin/env python3

import numpy as np
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate

import util
import random
import clifford_t_estim
import estimate

def fixed_probability_circuit(p, threads, print_round_by_round):
    #here we make a circuit of the form U U^dagger V(p) as described in equation (22) of https://arxiv.org/abs/2101.12223
    qubits = 40
    t_U = 16 # the U portion of the circuit has 16 T(theta) gates

    depth = 1000 # the U portion of the circuit has 1000 total gates
    count = 10000 # we search over 10000 randomly generated circuits

    rng = random.Random(1000) # initialize pseudo-random number generator

    measured_qubits = 8 # we are going to measure qubits 0-7 (inclusive)
    aArray = np.zeros(measured_qubits, dtype=np.uint8) # we are computing the probability for the outcome 00000000    

    effectiveT = 32
    beta = np.log2(4. - 2.*np.sqrt(2.));
    total_extent = np.power(2,beta*effectiveT) # phase gates have extent that varies with the phase so we can tune the T(theta) gates such that we get an extent that doesn't vary with p
    m = np.sqrt(total_extent)

    #params for Estimate
    epsTot = 0.1
    deltaTot = 0.001

    max_r  = 8# the maximum r we will allow we will search random circuits until we find one with r=8, we search for relatively small r as large ones can be efficiently computed by the Compute algorithm

    for i, U in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, count, t_U, rng)): # generate our random U circuits
        # our circuit is U U^dagger V(p)
        uDagger  = U.inverse()
        circ = CompositeCliffordGate()
        circ.gates = U.gates + uDagger.gates
        
        for j in range(measured_qubits):
            circ.gates.append(HGate(j))
            circ.gates.append(TGate(j))
            circ.gates.append(HGate(j))

        gates, controls, targets = util.convert_circuit_to_numpy_arrays(circ)

        #here we are interested in the performance without imposing the region c constraints detailed in appendix G of the manuscript
        #since we make no guarantees that the arrrays will not be changed by the code it is a good idea to copy them
        #particuarly since we want to pass the same ones to the estimate algorithm later
        d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, v, CH, AG = clifford_t_estim.compress_algorithm_no_region_c_constraints(qubits,
                                                                                                                                            measured_qubits,
                                                                                                                                            np.copy(gates),
                                                                                                                                            np.copy(controls),
                                                                                                                                            np.copy(targets),
                                                                                                                                            np.copy(aArray))
        
        if r <= max_r: # we have searched a bunch of circuits and found one with a low enough r that we're interested in it
            #now we compute the phases we need for our T(phi) and T(theta) gates
            
            phases = np.zeros(40, dtype=np.float64) # 40 = 16 + 16 + 8 for U, U^dagger and V
            
            phi = np.arccos(np.power(p, 1./16))*2  # it is easy to compute what phi you need for a given p

            #it is more complicated to compute what theta you need but the function is monotone so we can do interval bisection
            sqrt_remaining_extent_per_t = np.sqrt(4-2*np.sqrt(2))/np.power(np.sqrt(1-np.sin(phi)) + np.sqrt(1-np.cos(phi)), 1/4.)
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

            for k in range(32):
                phases[k] = theta
            for k in range(32, 40):
                phases[k] = phi
            return estimate.estimate_with_phases(epsTot, deltaTot, t, measured_qubits, r, v, m, CH, AG, phases, seed=None, threads=threads,print_round_by_round=print_round_by_round)
            
if __name__ == "__main__":
    prob = 0.2 # this is the outcome probability we build the circuit to obtain

    #you should change the threads parameter to the number of cores on your computer!
    #since the computation will take some time to run (around 20 mins on my computer) we print information every round of the Estimate algorithm
    #the computation will finish when eps* < epsTot    
    prob, eps = fixed_probability_circuit(prob, threads=4, print_round_by_round=True)
    print(prob, "+/-, ", eps)
