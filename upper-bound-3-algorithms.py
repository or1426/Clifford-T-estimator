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
random.seed(seed)

qubits = 4
measured_qubits = 2
depth = 10 # total number of gates per circuit
T = 3 # number of non-Clifford gates per circuit
count = 1000 # number of circuits sampled

alg_1_improvements = []
alg_2_improvements = []
alg_3_improvements = []
compress_solves = 0


#with open('dimension-r q={} w={} t={} depth={} count={}.txt'.format(qubits, measured_qubits, T,depth,count), "w") as f:
with open('/dev/null', "w") as f:
    for i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, count, T, rng=random)):
        if i % 1000 == 0:
            print(i)
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
                
                
        out = clifford_t_estim.upper_bound_alg_1(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        if type(out) == tuple:
           rows, c, r  = out
           if True: #rows - measured_qubits < 0:
               alg_1_improvements.append(rows - measured_qubits)
               
               
               out2 = clifford_t_estim.upper_bound_alg_2(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
               #nullity_upper_bound, final_t, r, d, z_region_rank, log_v);
               nullity_upper_bound, final_t, r, d,  v = out2                                
               alg_2_improvements.append(v + nullity_upper_bound - measured_qubits)                                             

               n, k,  commutativity_diagram_rank = clifford_t_estim.upper_bound_alg_3(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
               alg_3_improvements.append(T - commutativity_diagram_rank)
        else:
            compress_solves += 1
            
#print(alg_1_improvements)
#print(alg_2_improvements)
#print(alg_3_improvements)

#exit()

from matplotlib import pyplot as plt

#plt.hist(differences)
#plt.grid()
#plt.show()

#from matplotlib.colors import LogNorm

a1min, a1max = min(alg_1_improvements), max(alg_1_improvements)
a2min, a2max = min(alg_2_improvements), max(alg_2_improvements)
a3min, a3max = min(alg_3_improvements), max(alg_3_improvements)

a1_counts = np.zeros_like(range(a1min, a1max+1))
a2_counts = np.zeros_like(range(a2min, a2max+1))
a3_counts = np.zeros_like(range(a3min, a3max+1))


for a1,a2,a3 in zip(alg_1_improvements, alg_2_improvements, alg_3_improvements):
    a1_counts[a1-a1min] +=1
    a2_counts[a2-a2min] +=1
    a3_counts[a3-a3min] +=1
    

#norm=colors.LogNorm(vmin=0, vmax=np.max(grid)), 
fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(8,10),sharey="row")
fig.suptitle("qubits = {}, w = {}, t = {},\n depth = {}, count = {},\n compress solves = {}, seed = {} ".format(qubits, measured_qubits, T, depth, count, compress_solves, seed), fontsize=14)

ax[0][0].bar(range(a1min, a1max+1), a1_counts, align="center", color="green")
ax[0][0].tick_params(labelsize=14)
ax[0][0].set_ylabel("frequency", fontsize=14)
ax[0][0].set_xlabel("algorithm 1 improvement", fontsize=14)
ax[0][0].set_xticks(range(a1min, a1max+1))
ax[0][0].grid()

ax[0][1].bar(range(a2min, a2max+1), a2_counts, align="center", color="green")
ax[0][1].tick_params(labelsize=14)
#ax[0][1].set_ylabel("frequency", fontsize=14)
ax[0][1].set_xlabel("algorithm 2 improvement", fontsize=14)
ax[0][1].set_xticks(range(a2min, a2max+1))
ax[0][1].grid()

ax[0][2].bar(range(a3min, a3max+1), a3_counts, align="center", color="green")
ax[0][2].tick_params(labelsize=14)
#ax[0][2].set_ylabel("frequency", fontsize=14)
ax[0][2].set_xlabel("algorithm 3 improvement", fontsize=14)
ax[0][2].set_xticks(range(a3min, a3max+1))
ax[0][2].grid()

diffs12 = []
diffs32 = []
diffs31 = []

for a1,a2,a3 in zip(alg_1_improvements, alg_2_improvements, alg_3_improvements):
    diffs12.append(a1 - a2)
    diffs31.append(a3 - a1)
    diffs32.append(a3 - a2)

d12min, d12max = min(diffs12), max(diffs12)
d31min, d31max = min(diffs31), max(diffs31)
d32min, d32max = min(diffs32), max(diffs32)

d12_counts = np.zeros_like(range(d12min, d12max+1))
d31_counts = np.zeros_like(range(d31min, d31max+1))
d32_counts = np.zeros_like(range(d32min, d32max+1))

for d12,d31,d32 in zip(diffs12, diffs31, diffs32):
    d12_counts[d12-d12min] +=1
    d31_counts[d31-d31min] +=1
    d32_counts[d32-d32min] +=1

    
ax[1][0].bar(range(d12min, d12max+1), d12_counts, align="center", color="green")
ax[1][0].tick_params(labelsize=14)
ax[1][0].set_ylabel("frequency", fontsize=14)
ax[1][0].set_xlabel("alg 1 improvement -\nalg 2 improvement", fontsize=14)
ax[1][0].set_xticks(range(d12min, d12max+1))
ax[1][0].grid()

ax[1][1].bar(range(d31min, d31max+1), d31_counts, align="center", color="green")
ax[1][1].tick_params(labelsize=14)
#ax[1][1].set_ylabel("frequency", fontsize=14)
ax[1][1].set_xlabel("alg 3 improvement -\nalg 1 improvement", fontsize=14)
ax[1][1].set_xticks(range(d31min, d31max+1))
ax[1][1].grid()

ax[1][2].bar(range(d32min, d32max+1), d32_counts, align="center", color="green")
ax[1][2].tick_params(labelsize=14)
#ax[1][2].set_ylabel("frequency", fontsize=14)
ax[1][2].set_xlabel("alg 3 improvement -\nalg 2 improvement", fontsize=14)
ax[1][2].set_xticks(range(d32min, d32max+1))
ax[1][2].grid()

plt.show()
