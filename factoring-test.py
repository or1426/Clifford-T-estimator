#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome, PauliZProjector
from stabstate import StabState
import constants
from cliffords import SGate, CXGate, CZGate, HGate, CompositeGate, SwapGate
import itertools                    
import util
import random
import qk

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
    qubits, depth, N = 10, 10, 100

    #our algorithm has two phases
    #first we "fix" the state so that the first column of the F matrix is of the form 10...0 (1 followed by all zeros)
    #then we just chop off the first qubit from our ch-form
    
    #we count how many times the phase was zero, because we don't really care about this case (the whole state-vector is just zero)
    zero = 0
    #how many times our "fixed" state matches the original one
    eq12 = 0
    #and doesn't
    neq12 = 0
    #how many times our "fixed" and chopped down state matches the original
    eq13 = 0
    #and doesn't
    neq13 = 0
    
    for i, (vector, circuit) in enumerate(zip(random.choices(list(itertools.product(range(2), repeat=qubits)), k=N), util.random_clifford_circuits(qubits=qubits, depth=depth, N=N))):
        #vector is a randomly length n vector of bits (used as the initial s vector of our state)
        #circuit is a length "depth" list of random Clifford gates
        
        #if i % 100 == 0:
        print(i)
        #form an projector onto the 0 state of the zeroth qubit
        projector = PauliZProjector(0,0)
        #form a state by passing our initial state through the random Clifford circuit and then apply the projector
        state = StabState.basis(s=vector) | circuit | projector

        #we don't do anything else if it turned out our state was orthogonal to the support of the projector
        if abs(state.phase) > 10e-10:
            #compute the inner product with all 2**n computational basis states
            reconstructed_vector1 = np.zeros(2**qubits, dtype=np.complex)
            for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
                reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)

            #apply the first part of our algorithm
            #attempt to simplify the state so the first column of F is 10...0
            #empirically this always works - but I have not proved this!!!
            factoring.gaussian_eliminate(state)
            print(state.tab())
            #compute the inner products again to make sure we only did "allowed" operations
            #i.e. we should have changed the CH-form but not any of the inner products
            reconstructed_vector2 = np.zeros(2**qubits, dtype=np.complex)
            for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
                reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
            
            if eq(reconstructed_vector1, reconstructed_vector2):
                eq12 += 1
            else:
                neq12 += 1
            #now delete the zeroth qubit from our CH-form (delete the zeroth row and column of F, G and M, as well as the zeroth element of g, v, s)
            state2 = state.delete_qubit(0)
            #compute the computational-basis inner products again
            #note that we are still using a length 2**n vector
            reconstructed_vector3 = np.zeros(2**(qubits), dtype=np.complex)
                
            for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
                #only update elements of the vector which correspond to the half of the CB vectors with qubit zero in state zero
                if t[projector.target] == projector.a:
                    t2 = t[:projector.target] + t[projector.target+1:]
                    reconstructed_vector3[i] = MeasurementOutcome(np.array(t2, dtype=np.uint8)).apply(state2)
                    
            if eq(reconstructed_vector1, reconstructed_vector3):
                eq13 += 1
            else:
                neq13 += 1
                
        else:
            zero += 1

    print("zeros:", zero)
    print("1 == 2: ", eq12)
    print("1 != 2: ", neq12)
    print("1 == 3: ", eq13)
    print("1 != 3: ", neq13)
