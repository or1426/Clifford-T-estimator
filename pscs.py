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
import qk

from numpy import linalg

if __name__ == "__main__":
    np.set_printoptions(linewidth=128) #makes printing long matrices a bit better sometimes
    
    #produce the 3 qubit |000>
    state1 = CHState.basis(N=3)

    #produce the 2 qubit |01>
    state2 = CHState.basis(s=[0,1])

    #produce the 3 qubit |111>
    state3 = CHState.basis(s=[1,1,1])

    #print state1 in the form N F G M gamma v s w
    #where N is number of qubits, F, G, M and NxN matrices, gamma, v and s are column vectors and w is a complex number
    #looks like
    """
    3 100 100 000 0 0 0 (1+0j)
      010 010 000 0 0 0
      001 001 000 0 0 0
    """
    print(state1)
    #the CHState.tab method prints out the same information as a table
    #i.e. with column headings indicating which block is which
    print(state1.tab())

    #create a Hadamard gate on qubit 1
    h = HGate(1)
    
    #create a cnot with target 0 and control 1
    cnot = CXGate(0, 1)

    #apply h and cnot to state2 (in that order)
    state2 = cnot.apply(h.apply(state2))

    #we can overlaps
    #for example here <01|state2>
    #should be -1/sqrt(2)
    overlap = MeasurementOutcome([1,1]).apply(state2)
    print(overlap)
    
    #this is kinda hard to read so we can also write application in a way that looks more like a circuit
    #SGate(1) just produces an S gate applied to qubit 1
    #it is here just to demonstrate that you don't need to construct the gates in advance
    state1 = state1 | h | SGate(1) | cnot

    #this produces the state  (|0>|0> + i |1>|1>)|0>/sqrt(2)
    print(state1)

    #we can verify this computation by computing the overlaps with each computational basis state
    reconstructed_vector = np.zeros(2**(state1.N), dtype=np.complex) # we give the dtype otherwise numpy throws away the imaginary part
    for i, t in enumerate(itertools.product([0,1], repeat=state1.N)):
        reconstructed_vector[i] = state1 | MeasurementOutcome(np.array(t, dtype=np.uint8)) 

    #should be (very close to) [0.70710678, 0., 0., 0., 0., 0., 0.70710678j 0.] #note numpy writes complex numbers like 0.+0.70710678j
    print(reconstructed_vector) 
    
    #we can project our state onto the positive eigenspace of Z0 (i.e. multiply by |0><0| * I * I)
    #the first argument of the PauliZProjector is the target qubit, the second is the power a in P = (I + (-1)^a Z)/2,
    state1 = state1 | PauliZProjector(0,0)
    print(state1)
    
    #and compute the overlaps again
    reconstructed_vector = np.zeros(2**(state1.N), dtype=np.complex)
    for i, t in enumerate(itertools.product([0,1], repeat=state1.N)):
        reconstructed_vector[i] = state1 | MeasurementOutcome(np.array(t, dtype=np.uint8)) 

    #should be (very close to) [0.70710678, 0., 0., 0., 0., 0., 0 0.]
    print(reconstructed_vector) 

