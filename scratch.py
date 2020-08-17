#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome
from chstate import CHState
import constants
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate
import itertools                    
import util
import random
import magicsim
import pickle
import factoring
import math
import qk

import cPSCS

from matplotlib import pyplot as plt
from multiprocessing import Pool

#import qk
#from agstate import AGState
#import qiskit
#from qiskit import QuantumCircuit
#from qiskit.providers.aer import QasmSimulator
#from qiskit import Aer

import time

def basisTuple(n, *args):
    v = [0]*n
    for arg in args:
        v[arg] = 1
    return tuple(v)

def eq(a,b,eps=1e-10):
    return linalg.norm(a - b) < eps
    

from numpy import linalg

if __name__ == "__tests-test__":
    cPSCS.tests()


if __name__ == "__main__":

    n = 1
    depth = 1000
    qubits_range = np.array(range(2,63), dtype=np.int)
    bravyi_times = np.zeros_like(qubits_range, dtype=np.float)
    my_times = np.zeros_like(qubits_range, dtype=np.float)
    repeats=100
    for i,q in enumerate(qubits_range):
        for circ in util.random_clifford_circuits(q, depth,n):
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
        
            v1, v2 = cPSCS.time_equatorial_prods(q, 543,repeats, gateArray,controlArray,targetArray)
            bravyi_times[i] += v1
            my_times[i] += v2
            print(q, v1,v2)
    plt.plot(qubits_range, (bravyi_times/(repeats*n))*1000, label="Bravyi algorithm")
    plt.plot(qubits_range, (my_times/(repeats*n))*1000, label="My algorithm")
    plt.legend()
    plt.grid()
    plt.show()
    
            

if __name__ == "__multi__":
    qubits = 16
    print(qubits)
    circs = 1
    depth = 5000
    t= 12
    print("t=", t)
    #magic_samples = 1000
    #equatorial_samples = 1000
    #seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(8)] + [0 for _ in range(8)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    epss = [0.1, 0.01, 0.001]
    #epss = [0.0001]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    random.seed(121)
    threads = 10
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
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

        for i, (a,m) in enumerate(zip(aArray, mask)):
            if m:
                circ | PauliZProjector(target=i, a=a)
        d_time = time.monotonic_ns()
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        qk_time += time.monotonic_ns() - d_time
        print(qk_val)
        
        for eps in epss:
            print(eps)
            meps = 0.5*eps
            magic_samples = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + meps) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
            equatorial_samples = int(round((((qk_val + meps)/(eps-meps))**2)*(-np.log(delta))))
            for ratio in range(1,9):
                meps2 = ratio*eps/10.
                magic_samplesr = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + meps2) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
                equatorial_samplesr = int(round((((qk_val + meps2)/(eps-meps2))**2)*(-np.log(delta))))

                if 0 < magic_samplesr*equatorial_samplesr < magic_samples*equatorial_samples:
                    magic_samples = magic_samplesr
                    equatorial_samples = equatorial_samplesr

                    
            def run_algorithm_1(seed):
                d_time = time.monotonic()
                v1 = cPSCS.magic_sample_1(qubits,magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
                s1_time = time.monotonic() - d_time

                return (v1, s1_time)

            print(magic_samples)
            print(equatorial_samples)

            with Pool(threads) as p:
                ans = p.map(run_algorithm_1, [random.randrange(1,10000) for _ in range(threads)])
                total_time = 0
                mean_val = 0
                
                for val, time_val in ans:
                    mean_val += val/threads
                    total_time += time_val
                    
                print(ans)
                print(mean_val, total_time)
                print("-----------------------------------")
        
if __name__ == "__single-process__":
    # do some simulations make some graphs
    qubits = 16
    print(qubits)
    circs = 1
    depth = 5000
    t= 10
    print("t=", t)
    #magic_samples = 1000
    #equatorial_samples = 1000
    seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(8)] + [0 for _ in range(8)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    #epss = [0.1, 0.01,0.001]
    epss = [0.1]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    random.seed(14321)
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
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

        for i, (a,m) in enumerate(zip(aArray, mask)):
            if m:
                circ | PauliZProjector(target=i, a=a)
        d_time = time.monotonic_ns()
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        qk_time += time.monotonic_ns() - d_time
        print(qk_val)
        
        #magic_samples = int(round(((2*(math.pow(2,gamma*t/2)+qk_val)**2)/((math.sqrt( (1+relative_eps)*qk_val) - math.sqrt(qk_val))**2))/(math.log(delta/(2*relative_eps*relative_eps*qk_val*qk_val)))))
        #equatorial_samples = int(round(((((1+relative_eps))/(relative_eps))**2)*(-np.log(delta))))
        for eps in epss:
            print(eps)
            meps = 0.9*eps
            magic_samples = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + eps) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
            equatorial_samples = int(round((((qk_val + meps)/(eps-meps))**2)*(-np.log(delta))))
            print(magic_samples)
            print(equatorial_samples)
            #print(qk_val)
            d_time = time.monotonic_ns()
            
            v1 = cPSCS.magic_sample_1(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            s1_time += time.monotonic_ns() - d_time
            print(v1)
            print(s1_time/1e9)
            
            d_time = time.monotonic_ns()
            v2 = cPSCS.magic_sample_2(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            s2_time += time.monotonic_ns() - d_time
            
            print(v2)
            # print(s2_time/1e9)
            
            print()
            print(qk_time/1e9, s1_time/1e9,s2_time/1e9)
            print("-------------------")


if __name__ == "__equatorial_prods_test__":
    qubits = 20
    n = 1000
    depth = 100
    equatorial_qubits_num = 10
    computational_qubits_num = qubits-equatorial_qubits_num
    good = 0
    bad = 0
    for i, circ in enumerate(util.random_clifford_circuits(qubits, depth,n)):
        print(i)
        #circ = CompositeCliffordGate([HGate(2)])
        computational_qubits = random.sample(range(qubits), computational_qubits_num);
        print(computational_qubits)
        equatorial_qubits = [n for n in range(qubits) if not n in computational_qubits]
        
        computational_qubits_mask = np.zeros(qubits, dtype=np.uint8)
        computational_qubits_mask[computational_qubits] = np.uint8(1)
        computational_state = np.random.randint(0,2, qubits, dtype=np.uint8)*computational_qubits_mask
        
        equatorial_qubits_mask = np.zeros(qubits, dtype=np.uint8)
        equatorial_qubits_mask[equatorial_qubits] = np.uint8(1)

        equatorial_state = np.random.randint(0, 2, (equatorial_qubits_num,equatorial_qubits_num), dtype=np.uint8)
        equatorial_state = ((equatorial_state+equatorial_state.T)//2) % np.uint8(2) # have to be symmetric
        equatorial_state[np.diag_indices_from(equatorial_state)] = np.random.randint(0,4,equatorial_qubits_num, dtype=np.uint8)
        #equatorial_state = np.array([[1,1],[1,3]], dtype=np.uint8)
        
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

        n, F, G, M, g, v, s, phase = cPSCS.apply_gates_to_basis_state_project_and_reduce(qubits, gateArray, controlArray, targetArray, computational_state, computational_qubits_mask)
        state1 = CHState(n, F,G,M,g,v,s,phase)
        ans1 = cPSCS.equatorial_inner_product_wrapper2(equatorial_qubits_num, F, G, M, g, v, s, phase, equatorial_state).conjugate()
        
        n1, F1, G1, M1, g1, v1, s1, phase1 = cPSCS.partial_equatorial_inner_product_wrapper(qubits, gateArray, controlArray, targetArray, equatorial_qubits_mask, equatorial_state)
        state2 = CHState(n1, F1, G1, M1, g1, v1, s1, phase1)
        ans2 = cPSCS.measurement_overlap_wrapper2(computational_qubits_num, F1, G1, M1, g1, v1, s1, phase1, computational_state[computational_qubits_mask==1])
        if abs(ans1 - ans2) < 1e-8:
            good += 1
        else:
            bad += 1
            print(i)
            print(circ)
            print(computational_qubits_mask)
            print(computational_state)
            print(equatorial_state)
            print(state1.tab())
            print(state2.tab())
            print(ans1, ans2)
            print("--------------")
            pyState= CHState.basis(4)
            HGate(2).applyCH(pyState)
            print(pyState.tab())
            print("--------------")
            break
            
    print(good, bad)

if __name__ == "__alg-test__":
    qubits = 9
    n = 1
    depth =5000
    t_max=5
    magic_samples = 1000
    equatorial_samples = 1000
    seed = 3643
    #ps = np.zeros(n, dtype=np.complex)
    good = 0
    bad = 0
    meh = 0
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, n, t_max)):
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

        mask = np.random.randint(0,2,qubits, dtype=np.uint8)
        aArray = np.random.randint(0,2, qubits, dtype=np.uint8)*mask

        
            
        v1 = cPSCS.magic_sample_1(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))

        v2 = cPSCS.magic_sample_2(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))

        for i, (a,m) in enumerate(zip(aArray, mask)):
            if m:
                circ | PauliZProjector(target=i, a=a)
        
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        print(qk_val)
        print(v1)
        print(v2)
        #print(circ)
        print()
        #if abs(qk_val - v1) > 0.2 or abs(qk_val - v2) > 0.2:
        #    bad += 1            
        #else:
        #    good += 1
        #print(good, bad, meh)

if __name__ == "__measurement-test__":
    qubits = 100
    n = 1000
    depth=500
    good = 0
    bad = 0
    for i, circ in enumerate(util.random_clifford_circuits(qubits, depth, n)):
        print(i)
        a = np.random.randint(0,2, qubits, dtype=np.uint8)
        pyState = CHState.basis(qubits)
        circ.applyCH(pyState)        
        py = MeasurementOutcome(a).applyCH(pyState)

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
        #c = cPSCS.measurement_overlap_wrapper(qubits, gateArray, controlArray, targetArray, a)
        n, F, G, M, g, v, s, phase = cPSCS.apply_gates_to_basis_state_project_and_reduce(qubits, gateArray, controlArray, targetArray, a, np.ones_like(a))
        if abs(py - phase)  > 1e-10:
            print(py, phase)
            
            bad += 1
            break
        else:
            good += 1
    print(good, bad)
            

    
if __name__ == "__t__":
    qubits = 50
    n = 1
    depth = 100
    magic_samples = 1000
    equatorial_samples = 1000
    t_max = 20
    seed = 45
    #ps = np.zeros(n, dtype=np.complex)

    for  i, (T,circ) in enumerate(util.random_clifford_circuits_with_T(qubits, depth, n)):
        if(T > t_max):
            print("terr")
        else:
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

            mask = np.random.randint(0,2, qubits, dtype=np.uint8)
            while mask.sum() == qubits:
                mask = np.random.randint(0,2, qubits, dtype=np.uint8)
            aArray = np.random.randint(0,2, qubits, dtype=np.uint8) * mask

            v1 = cPSCS.magic_sample_1(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            v2 = cPSCS.magic_sample_2(qubits,magic_samples, equatorial_samples, seed,  gateArray, controlArray, targetArray, aArray, mask)

            print(v1)
            print(v2)
            print()

if __name__ == "__t__":
    qubits = 4

    good = 0
    bad = 0
    for  i, circ in enumerate([CompositeCliffordGate([CXGate(2, 3), HGate(3)])]):
        mask = np.array([1, 1, 1, 0],dtype=np.uint8)
        aArray = np.array([1, 0, 1, 0],dtype=np.uint8)        
        print(i)
        print("mask", mask)
        print("aaaa", aArray)
        pyState = CHState.basis(qubits)
        pyState = pyState | circ

        for q, m, a in zip(reversed(range(qubits)), reversed(mask), reversed(aArray)):            
            if m == 1:
                print("py deleting ", q)
                if a == 1:
                    x = XGate(q)
                    pyState | x                                 
                
                proj = PauliZProjector(target=q, a=0)
                pyState | proj
        for q, m, a in zip(reversed(range(qubits)), reversed(mask), reversed(aArray)):
            if m == 1:
                factoring.reduce_column(pyState,q)
                pyState = pyState.delete_qubit(q)
        
        gateArray = np.zeros(len(circ.gates), dtype=np.uint8)
        controlArray = np.zeros(len(circ.gates), dtype=np.uint)
        targetArray = np.zeros(len(circ.gates), dtype=np.uint)

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

        n, F, G, M, g, v, s, phase = cPSCS.apply_gates_to_basis_state_project_and_reduce(qubits, gateArray, controlArray, targetArray, aArray, mask)

        cState = CHState(n, F,G,M,g,v,s,phase)
        print(pyState.N, cState.N)
        reconstructed_vector1 = np.zeros(2**(pyState.N), dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=pyState.N)):
            reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).applyCH(pyState)
        reconstructed_vector2 = np.zeros(2**(cState.N), dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=cState.N)):
            reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).applyCH(cState)

        
        if eq(reconstructed_vector1, reconstructed_vector2):
            good += 1
        else:
            bad += 1
            print(circ)
            print(pyState.tab())
            print(cState.tab())
            break


if __name__ == "__factoring-test__":
    qubits = 10
    n = 1000
    depth = 1000
    good = 0
    bad = 0
    for  i, circ in enumerate(util.random_clifford_circuits(qubits, depth, n)):        
        mask = np.random.randint(0,2, qubits, dtype=np.uint8)
        while mask.sum() == qubits:
            mask = np.random.randint(0,2, qubits, dtype=np.uint8)
        
        aArray = np.random.randint(0,2, qubits, dtype=np.uint8) * mask
        
        
        print(i)
        pyState = CHState.basis(qubits)
        pyState = pyState | circ

        for q, m, a in zip(reversed(range(qubits)), reversed(mask), reversed(aArray)):            
            if m == 1:
                if a == 1:
                    x = XGate(q)
                    pyState | x                                 
                
                proj = PauliZProjector(target=q, a=0)
                pyState | proj
                if abs(pyState.phase) < 1e-10:
                    print("sketches")
                    print("good = {}, bad = {}".format(good,bad))
                else:
                    factoring.reduce_column(pyState,q)
                pyState = pyState.delete_qubit(q)
                
                
        
        gateArray = np.zeros(depth, dtype=np.uint8)
        controlArray = np.zeros(depth, dtype=np.uint8)
        targetArray = np.zeros(depth, dtype=np.uint8)

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

        n, F, G, M, g, v, s, phase = cPSCS.apply_gates_to_basis_state_project_and_reduce(qubits, gateArray, controlArray, targetArray, aArray, mask)

        cState = CHState(n, F,G,M,g,v,s,phase)

        reconstructed_vector1 = np.zeros(2**(pyState.N), dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=pyState.N)):
            reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).applyCH(pyState)
        reconstructed_vector2 = np.zeros(2**(cState.N), dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=cState.N)):
            reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).applyCH(cState)

        
        if eq(reconstructed_vector1, reconstructed_vector2):
            good += 1
        else:
            bad += 1
            print(circ)
            print(pyState.tab())
            print(cState.tab())
            break

    print("good = {}, bad = {}".format(good,bad))

if __name__ == "__test-clifford-unitaries__":
    qubits = 10
    n = 100
    depth = 100
    good = 0
    bad = 0
    for  i, circ in enumerate(util.random_clifford_circuits(qubits, depth, n)):
        print(i)
        pyState = CHState.basis(qubits)
        pyState = pyState | circ
        
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

        n, F, G, M, g, v, s, phase = cPSCS.apply_gates_to_basis_state(qubits, gateArray, controlArray, targetArray)
        cState = CHState(n, F,G,M,g,v,s,phase)
        if cState == pyState:
            good += 1
        else:
            bad += 1

    print("good = {}, bad = {}".format(good,bad))

    
if __name__ == "__timing__":
    #qubits = 40
    qs = np.array(range(2,64))
    #py_ts = np.zeros_like(qs, dtype=np.float)
    c_ts = np.zeros_like(qs, dtype=np.float)
    #conv_ts = np.zeros_like(qs, dtype=np.float)
    n = 1000
    depth = 10000
    #
    #for i, circ in enumerate([CompositeCliffordGate([HGate(0)])]):
    #CX(0, 2), CX(0, 2), CX(4, 3)]
    #for  i, circ in enumerate(util.random_clifford_circuits(qubits, depth, n)):    
    good = 0
    bad = 0
    #
    #for i, circ in enumerate([CompositeCliffordGate([HGate(4),HGate(4)])])
    #
    #
    #for i, circ in enumerate([CompositeCliffordGate([HGate(1), SGate(1), SGate(1), HGate(1)])]):
    for i,qubits in enumerate(qs):
        print(qubits)
        for  _, circ in enumerate(util.random_clifford_circuits(qubits, depth, n)):            
            #pyState = CHState.basis(qubits)
            #delta = time.monotonic()
            #pyState = pyState | circ
            #py_ts[i] += (time.monotonic() - delta)

            delta = time.monotonic()
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
            #conv_ts[i] += (time.monotonic() - delta)
            delta = time.monotonic()
            F, G, M, g, v, s, phase = cPSCS.apply_gates_to_basis_state(qubits, gateArray, controlArray, targetArray)
            c_ts[i] += (time.monotonic()-delta)
            
            cState = CHState(qubits, F,G,M,g,v,s,phase)
    with open("64bit_big_c_timing.pkl", "wb") as f:
        pickle.dump((qs,c_ts), f)

                
        

if __name__ == "__euqatorial-test__":
    qubits = 2
    n = 100
    depth = 10
    #ps = np.zeros(n, dtype=np.complex)
    good = 0
    bad = 0
    for  i, circ in enumerate(util.random_clifford_circuits(qubits, depth, n)):
        #circ = CompositeCliffordGate([HGate(1), CXGate(control=1,target=0)])
        print(i)
        #equatorial_state = np.array([[3,1],[1,3]])
        equatorial_state = np.random.randint(0, 2, (qubits,qubits), dtype=np.uint8)
        equatorial_state = ((equatorial_state+equatorial_state.T)//2) % np.uint8(2) # have to be symmetric
        equatorial_state[np.diag_indices_from(equatorial_state)] = np.random.randint(0,4,qubits, dtype=np.uint8)
        pyState = CHState.basis(qubits)
        pyState = pyState | circ

        py = pyState.equatorial_inner_product(equatorial_state)

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

        c = cPSCS.equatorial_inner_product_wrapper(qubits, gateArray, controlArray, targetArray, equatorial_state)
        N, F, G, M, g, v, s, phase = cPSCS.partial_equatorial_inner_product_wrapper(qubits, gateArray, controlArray, targetArray, np.ones(qubits, dtype=np.uint8), equatorial_state)
        if abs(py - c) < 1e-10 and abs(c - phase.conjugate()) < 1e-10:
            good += 1
        else:
            bad += 1
            print(py)
            print(c)
            print(phase.conjugate())
            print()
            break
        
    print(good, bad)
if __name__ == "__timing__":
    tests = 100
    depth = 1000
    inner_prods = 100
    qs = np.array(range(2,3))
    ts = np.zeros_like(qs, dtype=np.float)
    
    for i, qubits in enumerate(qs):
        print(qubits)
        for  circ in util.random_clifford_circuits(qubits, depth, tests):
            state = CHState.basis(qubits)
            state = state | circ
        
            for _ in range(inner_prods):
                equatorial_state = np.random.randint(0, 2, (qubits,qubits), dtype=np.uint8)
                equatorial_state = ((equatorial_state+equatorial_state.T)//2) % np.uint8(2) # have to be symmetric
                delta = time.monotonic()
                p1 = state.equatorial_inner_product(equatorial_state)
                ts[i] += time.monotonic() - delta
        ts[i] *= (1000/(tests*inner_prods))
            
    print(qs)
    print(ts)
    with open("full_inner_prod_data2.pkl", "wb") as f:
        pickle.dump((qs,ts), f)

if __name__ == "__test__":
    qubits =63
    tests = 100
    depth = 1000
    inner_prods = 10
    good = 0
    bad = 0
    t = 0
    
    for i, circ in enumerate(util.random_clifford_circuits(qubits, depth, tests)):
        print(i)
        state = CHState.basis(qubits)
        #state.s = np.randint(0, 2, qubits, dtype=np.uint8)
        #state.v = np.randint(0, 2, qubits, dtype=np.uint8)

        state = state | circ

        
        for _ in range(inner_prods):
            equatorial_state = np.random.randint(0, 2, (qubits,qubits), dtype=np.uint8)
            equatorial_state = ((equatorial_state+equatorial_state.T)//2) % np.uint8(2) # have to be symmetric

            #delta = time.monotonic()
            p1 = state.equatorial_inner_product(equatorial_state)

            if p1:
                good += 1
            else:
                bad += 1
            #delta = time.monotonic() - delta
            #t += p1
        
        #p2 = 0
        #for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
        #    #reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
        #    x = np.array(t)
        #    p2 +=  MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state).conjugate() * ((1j)**(np.int(x @ equatorial_state @ x)))

        #p2 = p2 / (2**(qubits/2))

        #print(p1, p2)
        #print(abs(p1-p2) < 1e-10)
        #if abs(p1-p2) > 1e-10:
        #    break
    #print((t/(tests*inner_prods))*1000)
    print("good = {}, bad = {}".format(good,bad))

    
if __name__ == "__exponential-sum-test__":
    qubits = 5
    tests = 1000
    g = 0
    b = 0
    
    for _ in range(tests):
        M = np.random.randint(0,2,(qubits,qubits), dtype=np.uint8)
        L = np.random.randint(0,2,qubits, dtype=np.uint8)

        s1 = util.z2ExponentialSum(M, L)
        s2 = util.slowZ2ExponentialSum(M,L)
        if s1 != s2:
            print("Error:" , s1, s2 )
            b += 1
        else:
            g += 1

    print("Tests: ", tests, "Passes: ", g, "Fails: ", b)


        
    

if __name__ == "__qk-test__":
    qubits = 10
    tests = 1000
    depth = 10

    for i, circ in enumerate(util.random_clifford_circuits(qubits, depth, tests)):
        print(i)
        state = CHState.basis(qubits)
        #state.s = np.randint(0, 2, qubits, dtype=np.uint8)
        #state.v = np.randint(0, 2, qubits, dtype=np.uint8)

        state = state | circ

        reconstructed_vector = np.zeros(2**qubits, dtype=np.complex)
        for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
            reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
    
        
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros_like(state.s), circ)

        #print(reconstructed_vector)
        #print(qk_vector)
        if not eq(reconstructed_vector, qk_vector):
            print("error")



if __name__ == "__main2__":
    qubits = 2
    circs = 1
    depth = 10
    magic_samples = 10
    cb_states = 10

    #circ = CompositeGate([SGate(2), HGate(7), TGate(4), HGate(5), CZGate(4, 2), HGate(1), CXGate(1, 6), CZGate(9, 3), CXGate(5, 1), HGate(2)])
    circ = CompositeGate([TGate(1), SGate(0), HGate(0)])
    print(circ)
    trivialSim = magicsim.TrivialSimulator(qubits, circ)
    magicSim = magicsim.MagicSimulator(qubits, circ)
    
    t = len([g for g in circ.gates if isinstance(g, TGate)])

    print("Trivial:")
    print((CHState.basis(qubits+t) | HGate(2) |trivialSim.circ).tab())
    print("Magic:")
    print(magicSim.chState.tab())
    print(magicSim.agState.tab())
    print(magicSim.agState.stabs())
    print(magicSim.agState.destabs())

    ys = np.array([1], dtype=np.uint8) 
    trivialState = trivialSim.magic_sample(ys)
    magicState = magicSim.magic_sample(ys)

    
    #cbState= np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
    cbState= np.array([1, 0, 1], dtype=np.uint8)
    o1 = MeasurementOutcome(cbState).apply(trivialState)
    o2 = MeasurementOutcome(cbState).apply(magicState)
    if abs(o1-o2) > 1e-10:
        print(ys)
        print(cbState)
        print("Trivial sim: ", o1)
        print("Magic sim: ", o2)
        
        clifCirc = []
        t2 = 0

        for gate in circ.gates:
            if isinstance(gate, TGate):
                target = gate.target
                clifCirc.append(CXGate(control=target, target=qubits+t2))
                t2 += 1
            else:
                clifCirc.append(gate)

        clifCirc = CompositeCliffordGate([HGate(qubits+p) for p in range(t)]) | CompositeCliffordGate([SGate(qubits+p) for p in range(t) if ys[p] == 1]) | CompositeCliffordGate(clifCirc)
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits+t, np.zeros(qubits+t,dtype=np.uint8), clifCirc)
        for index, ts in enumerate(itertools.product([0,1], repeat=qubits+t)):
            if np.equal(ts, cbState).all():
                print("qiskit:", qk_vector[index])

if __name__ == "__magic-test__":
    qubits = 20
    circs = 100
    depth = 200

    magic_samples = 10

    cb_states = 10
    
    for i, (t, circ) in enumerate(util.random_clifford_circuits_with_T(qubits, depth, circs)):
        print(i)
        trivialSim = magicsim.TrivialSimulator(qubits, circ)
        magicSim = magicsim.MagicSimulator(qubits, circ)

        for m in range(magic_samples):
            ys = np.random.randint(0,2, t, np.uint8)
            trivialState = trivialSim.magic_sample(ys)
            magicState = magicSim.magic_sample(ys)
        
            for c in range(cb_states):
                cbState=np.random.randint(0,2, qubits+t, np.uint8)
                o1 = MeasurementOutcome(cbState).apply(trivialState)
                o2 = MeasurementOutcome(cbState).apply(magicState)
                if abs(o1-o2) > 1e-10:
                    print(circ)
                    print("error", i, m, c)
                    print("Trivial sim: ", o1)
                    print("Magic sim: ", o2)

                    clifCirc = []
                    t2 = 0

                    for gate in circ.gates:
                        if isinstance(gate, TGate):
                            target = gate.target
                            clifCirc.append(CXGate(control=target, target=qubits+t2))
                            t2 += 1
                        else:
                            clifCirc.append(gate)

                    clifCirc = CompositeCliffordGate([HGate(qubits+p) for p in range(t)]) | CompositeCliffordGate([SGate(qubits+p) for p in range(t) if ys[p] == 1]) | CompositeCliffordGate(clifCirc)
                    sim = qk.QiskitSimulator()
                    qk_vector = sim.run(qubits+t, np.zeros(qubits+t,dtype=np.uint8), clifCirc)
                    for index, ts in enumerate(itertools.product([0,1], repeat=qubits+t)):
                        if np.equal(ts, cbState).all():
                            print("qiskit:", qk_vector[index])
                    exit()
            
if __name__ == "__ag__":
    q = 1
    chstate = CHState.basis(q)
    agstate = AGState.basis(q)

    #circ = CompositeGate([HGate(0), CXGate(target=1, control=0)])
    circ = CompositeCliffordGate([HGate(0), SGate(0), HGate(0)])

    circ2 = HGate(0) | SGate(0) | TGate(0)
    circ2 | SGate(0)
    
    chstate | circ 
    agstate | circ
    #circ.apply(chstate)
    #circ.applyAG(agstate)

    print(agstate.tab())
    print(agstate.stabs())
    print(agstate.destabs())

    
    
    #circs = util.random_clifford_circuits(q,1000,100)
    

    #for circ in circs:
    #print(circ)
    #    m = MeasurementOutcome(np.random.randint(0,2, q, dtype=np.uint8))
    #    circ.applyAG(agstate)
    #    circ.apply(chstate)
    #    ag = m.applyAG(agstate)
    #    ch = abs(m.apply(chstate))
    #    if max([ag,ch]) > 1e-10 and abs(ag-ch)/max([ag,ch]) > 1e-10:
    #        print(ag, abs(ch))

        

    
    
        
if __name__ == "__s__":
    qubits = 10
    tests = 1
    depth = 100

    for circ in util.random_clifford_circuits(qubits, depth, tests):
        state = CHState.basis(qubits)
        #state.s = np.randint(0, 2, qubits, dtype=np.uint8)
        #state.v = np.randint(0, 2, qubits, dtype=np.uint8)

        state = state | circ

        print(state.G.T @ state.M % np.uint8(2))
        #k = 2
        #print(sum([state.G[j] * state.F[j][k] for j in range(qubits)]) %np.uint(2))
        
if __name__ == "__test__":

    print(-1)
    state = CHState.basis(3)
    print(state.tab())
    
    state = CHState.basis(3) | HGate(2) |  HGate(0)  | CXGate(target=1,control=0) | CXGate(target=2,control=1)
    print(state.tab())
    
    reconstructed_vector1 = np.zeros(2**3, dtype=np.complex)
    for i, t in enumerate(itertools.product([0,1], repeat=3)):
        reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
        
        
    F = np.array([[1,0,0],
                  [1,1,0],
                  [1,1,1]],dtype=bool)
    G = np.array([[1,1,0],
                  [0,1,1],
                  [0,0,1]],dtype=bool)
    M = np.array([[0,0,0],
                  [0,0,1],
                  [0,1,1]],dtype=bool)
    
    g = np.array([0,0,2], dtype=np.uint8)
    v = np.array([1,1,1], dtype=bool)
    s = np.array([0,0,0], dtype=bool)
    w = 1
    t1State = CHState(N=3, A=F, B=G, C=M, g=g, v=v,s=s,phase=w)
    
    
    reconstructed_vector2 = np.zeros(2**3, dtype=np.complex)
    for i, t in enumerate(itertools.product([0,1], repeat=3)):
        reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(t1State)
       
    print(reconstructed_vector1)
    print(reconstructed_vector2)

    print(linalg.norm(reconstructed_vector1 - reconstructed_vector2))
    backend = Aer.get_backend("statevector_simulator")
    circuit = QuantumCircuit(3)

    circuit.h(2)
    circuit.cx(target_qubit=1, control_qubit=2)
    circuit.h(0)
    circuit.cx(target_qubit=0, control_qubit=1)
    circuit.h(1)

    job = qiskit.execute(circuit, backend=backend,backend_options={"method": "statevector"})
    statevector = qk._rearange_state_vector(3, job.result().get_statevector(circuit))

    print(statevector)


if __name__ == "__derivs__":
    qubits = 3
    T = 3
    d1 = 10
    d2=10
    N = 10000    
    
    for circ1, circ2 in zip(util.random_clifford_circuits(qubits, depth=d1, N=N),util.random_clifford_circuits(qubits, depth=d2, N=N)):
        circ1 = HGate(0) | circ1
        states = {}
        for y in itertools.product(range(2), repeat=T):
            states[y] = CHState.basis(qubits+T)
            states[y] | circ1

            for t in range(T):
                states[y] | HGate(qubits+t)
            for i, val in enumerate(y):
                if val == 1:
                    states[y] | SGate(qubits+i)
            for t in range(T):
                states[y] | CXGate(control=t, target=qubits+t)

            states[y] | circ2

            #states[y] = states[y] | PauliZProjector(qubits, a=0)
            #states[y] = states[y] | PauliZProjector(qubits+1, a=0)
            #states[y] = states[y] | PauliZProjector(qubits+2, a=0)

        derivs = {}
        for t in range(T):
            derivs[(t, )] = (states[basisTuple(T,t)] - states[basisTuple(T)]) 

        
        for t1, t2 in itertools.product(range(T),repeat=2):
            derivs[(t1,t2)] = states[basisTuple(T, t1,t2)] - states[basisTuple(T, t1)] - states[basisTuple(T,t2)] + states[basisTuple(T)]


        y = [0]*T
        y[0] = 1
        y[1] = 1
        y[2] = 1
        y = tuple(y)


        state = CHState.basis(qubits+T)
        state | circ1
        for t in range(T):
            state | HGate(qubits+t)
        for i, val in enumerate(y):
            if val == 1:
                state | SGate(qubits+i)
        for t in range(T):
            state | CXGate(control=t, target=qubits+t)
        state | circ2
        #state | PauliZProjector(qubits, a=0)
        #state | PauliZProjector(qubits+1, a=0)
        #state | PauliZProjector(qubits+2, a=0)

        s = states[basisTuple(T)]

        for i1,i2 in itertools.product(range(T), repeat=2):
            if i1 == i2:
                if y[i1] == 1:
                    s = s + derivs[(i1,)]
            elif i1 > i2:
                if y[i1] == 1 and y[i2] == 1:
                    s = s + derivs[(i1,i2)]
        

        #factoring.gaussian_eliminate(state.F)
        if (s.F != state.F).any():
            print(circ1)
            print(circ2)
            print(state.F)
            print(s.F)

            reconstructed_vector1 = np.zeros(2**(qubits+T), dtype=np.complex)
            for i, t in enumerate(itertools.product([0,1], repeat=qubits+T)):
                reconstructed_vector1[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)

            reconstructed_vector2 = np.zeros(2**(qubits+T), dtype=np.complex)
            for i, t in enumerate(itertools.product([0,1], repeat=qubits+T)):
                reconstructed_vector2[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(s)

            print(reconstructed_vector1)
            print(reconstructed_vector2)
            print(eq(reconstructed_vector1, reconstructed_vector2))
            print(state.tab())
            print(s.tab())
            break



        
    
