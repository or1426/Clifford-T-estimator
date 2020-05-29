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
import factoring
import qk

from copy import deepcopy

def eq(a,b,e=1e-10):
    return linalg.norm(a - b) < e
    
def basisTuple(n, *args):
    v = [0]*n
    for arg in args:
        v[arg] = 1
    return tuple(v)
        
from numpy import linalg

if __name__ == "__main__":
    code = 3
    magic = 3
    qubits = code+magic
    #state = CHState.basis(code+magic) # 3 code and 3 magic qubits

    circ1 = HGate(1) | CXGate(target=0,control=1) | SGate(2) | HGate(2)
    circ2 = HGate(1) | CXGate(target=2,control=0) | HGate(0) | CZGate(target=0, control=1)
    #circ2 = CXGate(target=2,control=0) | CZGate(target=0, control=1)
    
    states = {} # form a dictionary, this will store the state we get out with each choice of y
    for y in itertools.product(range(2), repeat=magic): #we iterate over all 8 tuples (a,b,c) where a, b and c are 0 or 1
        states[y] = CHState.basis(code+magic)
        states[y] | circ1

        for t in range(magic): # each magic qubit gets an H
            states[y] | HGate(code+t)
        for t, val in enumerate(y): # if y_i = 1 we get an S 
            if val == 1:
                states[y] | SGate(code+t)
        for t in range(magic): # each magic qubit gets a cnot to its code qubit
            states[y] | CXGate(control=t, target=code+t)

        states[y] | circ2

        
        for i in range(qubits):
            if states[y].s[i] == 1 and states[y].v[i] == 1:
                states[y].s[i] = 0
                SGate(i).rightMultiplyC(SGate(i).rightMultiplyC(states[y]))

        
        #reconstructed_vector = np.zeros(2**qubits, dtype=np.complex)
        #for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
        #    reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(states[y])
    
        
        #sim = qk.QiskitSimulator()
        #qk_vector = sim.run(qubits, np.zeros_like(states[y].s), circ1 |HGate(3)|HGate(4)| HGate(5)|CXGate(3,0)|CXGate(4,1)| CXGate(5,2) |circ2)
        #print(reconstructed_vector)
        #print(qk_vector)

        #print(eq(reconstructed_vector,qk_vector))
        #break

    
    
        #maybe we want to try putting the projectors in here
        #states[y] = states[y] | PauliZProjector(code, a=0)
        #states[y] = states[y] | PauliZProjector(code+1, a=0)
        #states[y] = states[y] | PauliZProjector(code+2, a=0)
    
    for y in states.keys():
        print("y = ", y)
        print(states[y].tab())
        
    derivs = {}
    for t in range(magic): # compute all first derivatives
        derivs[(t, )] = (states[basisTuple(magic,t)] - states[basisTuple(magic)]) 

        
    for t1, t2 in itertools.product(range(magic),repeat=2): # compute all second derivatives
        derivs[(t1,t2)] = states[basisTuple(magic, t1,t2)] - states[basisTuple(magic, t1)] - states[basisTuple(magic,t2)] + states[basisTuple(magic)]
        
    #for key in derivs.keys():
    #    print(key)
    #    print(derivs[key])
    #now choose a y in {0,1}^magic
    #if there are 0, 1 or 2 ones, the approximation should be exact, if there are three we hope it still works
    y = [0]*magic
    y[0] = 1
    y[1] = 1
    y[2] = 1
    y = tuple(y)


    #compute the actual state given y
    state = CHState.basis(code+magic)
    state | circ1
    for t in range(magic):
        state | HGate(code+t)
    for t, val in enumerate(y):
        if val == 1:
            state | SGate(code+t)
    for t in range(magic):
        state | CXGate(control=t, target=code+t)
        
    state | circ2
    #if we did the projections above we should do them here
    #state | PauliZProjector(code, a=0)
    #state | PauliZProjector(code+1, a=0)
    #state | PauliZProjector(code+2, a=0)

    #compute the quadradic approximation of state(y)
    approx = states[basisTuple(magic)] # this is state(000)
    for i1,i2 in itertools.product(range(magic), repeat=2):
        if i1 == i2: 
            if y[i1] == 1:
                approx = approx + derivs[(i1,)]
        elif i1 > i2:
            if y[i1] == 1 and y[i2] == 1:
                approx = approx + derivs[(i1,i2)]
    print("Actual state:")
    print(state.tab())
    print("Quadratic approximation:")
    print(approx.tab())
    print("Difference")
    print((state-approx).tab())

    #where do the F matrices differ?
    print("Differences in F")
    print(np.uint8(state.F != approx.F))

    

    reconstructed_vector1 = np.zeros(2**(code+magic), dtype=np.complex)
    for i, t in enumerate(itertools.product([0,1], repeat=code+magic)):
        reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)

    reconstructed_vector2 = np.zeros(2**(code+magic), dtype=np.complex)
    for i, t in enumerate(itertools.product([0,1], repeat=code+magic)):
        reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(approx)
        
    #these are long so just print out the first elements
    print("First element of computational basis state vectors:")
    print("    Actual: ", reconstructed_vector1[0])
    print("    Quadradic approximaion: ",reconstructed_vector2[0])
    
