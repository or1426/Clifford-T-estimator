#! /usr/bin/env python3

import numpy as np
from chstate import CHState
from agstate import AGState
import constants
import gates
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate
import itertools                    
import util
import random
import matplotlib.cm as cm
import matplotlib.colors as colors

import clifford_t_estim
import random

seed = 1001 #random.randrange(0, 100000)
#print(seed)
random.seed(seed)

qubits = 4
measured_qubits = 2
depth = 6 # total number of gates per circuit
T = 3 # number of non-Clifford gates per circuit
count = 1000 # number of circuits sampled

alg_1_wins_circuit = None
alg_2_wins_circuit = None
alg_3_wins_circuit = None


def convert_circuit_to_arrays(circ):
    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)    
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
    return gateArray, controlArray, targetArray, aArray


while (alg_1_wins_circuit == None) or (alg_2_wins_circuit == None) or (alg_3_wins_circuit == None): # loop until we find an example when each wins
    for circ in util.random_clifford_circuits_with_bounded_T(qubits, depth, count, T, rng=random):
        gateArray, controlArray, targetArray, aArray = convert_circuit_to_arrays(circ)
        
        out = clifford_t_estim.upper_bound_alg_1(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        if type(out) == tuple: # if this is false then compress solved everything for us
            rows, c, r  = out
            alg_1_improvement = rows - measured_qubits
            nullity_upper_bound, final_t, r, d,  v = clifford_t_estim.upper_bound_alg_2(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
            alg_2_improvement = v + nullity_upper_bound - measured_qubits        
            n, k,  commutativity_diagram_rank = clifford_t_estim.upper_bound_alg_3(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
            alg_3_improvement = T - commutativity_diagram_rank

            if alg_1_wins_circuit == None:
                if alg_1_improvement > max(alg_2_improvement, alg_3_improvement):
                    alg_1_wins_circuit = circ
            if alg_2_wins_circuit == None:
                if alg_2_improvement > max(alg_1_improvement, alg_3_improvement):
                    alg_2_wins_circuit = circ
            if alg_3_wins_circuit == None:
                if alg_3_improvement > max(alg_2_improvement, alg_1_improvement):
                    alg_3_wins_circuit = circ

print("alg 1 circuit")
print(alg_1_wins_circuit)
print()
print("alg 2 circuit")
print(alg_2_wins_circuit)
print()
print("alg 3 circuit")
print(alg_3_wins_circuit)
print()


