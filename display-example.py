#! /usr/bin/env python3 

import numpy as np
import gates
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate
import itertools                    
import util
import random
import clifford_t_estim

def convert_circuit_to_arrays(circ, measured_qubits):
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



qubits, T, measured_qubits = 6, 3, 3

circ = CompositeGate([HGate(0), CZGate(4, 2), SGate(0), TGate(2), HGate(1), SGate(1), TGate(0), HGate(2), HGate(3), HGate(2), TGate(0), SGate(4), HGate(5), SGate(5), SGate(5), HGate(2), SGate(5), CZGate(0, 4), CXGate(2, 1), HGate(0), CXGate(5, 2), CZGate(1, 0), CXGate(3, 5), CZGate(4, 0), HGate(3), CXGate(5, 2), HGate(5), CXGate(4, 0)])
depth = len(circ.gates)
from stabtableau import GadgetizedStabState

print(circ)
state = GadgetizedStabState(qubits, circ)
print(state)
gateArray, controlArray, targetArray, aArray = convert_circuit_to_arrays(circ, measured_qubits)
n, k,  commutativity_diagram_rank = clifford_t_estim.upper_bound_alg_3(1, qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
alg_3_improvement = T - commutativity_diagram_rank
out = clifford_t_estim.upper_bound_alg_2(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
nullity_upper_bound, final_t, r, d,  v = out
alg_2_improvement = v + nullity_upper_bound - measured_qubits

                
rows, c, r  = clifford_t_estim.upper_bound_alg_1(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
alg_1_improvement = v + rows - measured_qubits
print("alg 1:", alg_1_improvement)
print("alg 2:", alg_2_improvement)
print("alg 3:", alg_3_improvement)
m_y = clifford_t_estim.slowly_compute_m_upper_bound(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
print("ub   :", m_y)
