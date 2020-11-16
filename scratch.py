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
from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
import sys
import qiskit

import cPSCS

from matplotlib import pyplot as plt
from multiprocessing import Pool

import time

def basisTuple(n, *args):
    v = [0]*n
    for arg in args:
        v[arg] = 1
    return tuple(v)

def eq(a,b,eps=1e-10):
    return linalg.norm(a - b) < eps
    

from numpy import linalg


if __name__ == "__qk_test_1__":
    qubitss = np.array(range(15,25))
    
    t = 30
    depth = 1000
    circs = 100
    measured_qubits = 10

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)
    #backend_options = {"max_parallel_threads": 1,}
    #backend = qk.Aer.get_backend("statevector_simulator", max_parallel_threads=1)
    #backend = Aer.get_backend("statevector_simulator")
    
    #backend = QasmSimulator(method='statevector', max_parallel_threads=1)
    #backend = StatevectorSimulator(*{"max_parallel_threads":6})

    #qk_times = []
    #comp_times = []
    for qubits in qubitss:        
        for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
            print(qubits,i)
            gateArray = np.zeros(depth, dtype=np.uint8)
            controlArray = np.zeros(depth, dtype=np.uint)
            targetArray = np.zeros(depth, dtype=np.uint)
            qkcirc = qiskit.QuantumCircuit(qubits)
            for j, gate in enumerate(circ.gates):
                if isinstance(gate, CXGate):
                    gateArray[j] = 88 #X
                    controlArray[j] = gate.control
                    targetArray[j] = gate.target
                    qkcirc.cx(target_qubit=gate.target, control_qubit=gate.control)
                elif isinstance(gate, CZGate):
                    gateArray[j] = 90 #Z
                    controlArray[j] = gate.control
                    targetArray[j] = gate.target
                    qkcirc.cx(target_qubit=gate.target, control_qubit=gate.control)
                elif isinstance(gate, SGate):
                    gateArray[j] = 115 #s
                    targetArray[j] = gate.target
                    qkcirc.s(gate.target)
                elif isinstance(gate, HGate):
                    gateArray[j] = 104 #h
                    targetArray[j] = gate.target
                    qkcirc.h(gate.target)
                elif isinstance(gate, TGate):
                    gateArray[j] = 116 # t
                    targetArray[j] = gate.target
                    qkcirc.t(gate.target)
                    
            for i  in range(measured_qubits):
                circ | PauliZProjector(target=i, a=aArray[i])
                    
            #print("starting qk sim")
            #print("1")
            sim = qk.QiskitSimulator()
            #print("2")
            d_time = time.monotonic()
            qk_vector = sim.run(qubits, np.zeros(qubits,dtype=np.uint8), circ)
            qk_time = time.monotonic - d_time
            #job = qiskit.execute(qkcirc, backend=backend, backend_options={"max_parallel_threads":6})
            #qk_vector = job.result().get_statevector(qkcirc)

            qk_val = qk_vector.conjugate() @ qk_vector

            #qk_times.append(qk_time)
            #print("ended qk sim")
            #print("starting CALCULATE sim")
            #print("3")
            d_time = time.monotonic()
            comp_val = cPSCS.calculate_algorithm(qubits, measured_qubits,gateArray, controlArray, targetArray, aArray)
            
            comp_time = time.monotonic() - d_time
            #print("4")
            #print("ended CALCULATE sim")
            #with open("qk_calculate_comparison.txt", "a") as f:
            print(qubits, qk_val, comp_val, qk_time, comp_time)
            
if __name__ == "__main__":
    qubits = int(sys.argv[1])
    t = 30
    depth = 1000
    circs = 100
    measured_qubits = 10

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        print(qubits,i)
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
                    
        for i  in range(measured_qubits):
            circ | PauliZProjector(target=i, a=aArray[i])
                    
        sim = qk.QiskitSimulator()

        d_time = time.monotonic()
        qk_vector = sim.run(qubits, np.zeros(qubits,dtype=np.uint8), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        qk_time = time.monotonic() - d_time
        
        d_time = time.monotonic()
        comp_val = cPSCS.calculate_algorithm(qubits, measured_qubits,gateArray, controlArray, targetArray, aArray)
        comp_time = time.monotonic() - d_time
            
        with open("qk_calculate_comparison_{}.txt".format(qubits), "a") as f:
            print(qubits, qk_val, comp_val, qk_time, comp_time, file=f)
            
            

if __name__ == "__compress-test__":
    qubits = 50
    t = 40
    depth = 100000
    circs = 10000
    measured_qubits = 10

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)
    
    data = []
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        print(i)
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
        data.append(cPSCS.v_r_info(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray))


    with open("v_r_data2.pkl", "wb") as f:
        pickle.dump(data, f)
                    
        
    
if __name__ == "__v_r__":
    qubits = 100
    t = 100
    depth = 1000
    circs = 100
    measured_qubits = qubits

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)
    

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

        cPSCS.print_v_r_info(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray)

if __name__ == "__test__":
    qubits = 10
    t = 10
    depth = 1000
    #circ = CompositeCliffordGate([TGate(1)])
    circ = list(util.random_clifford_circuits_with_bounded_T(qubits, depth, 1, t))[0]
    
    eps = 0.1
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    random.seed(131)
    seed = 111
    mask = np.array([1,0], dtype=np.uint8)
    aArray = np.zeros_like(mask)
    
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
    sim = qk.QiskitSimulator()
    qk_vector = sim.run(qubits, np.zeros(qubits), circ)
    qk_val = qk_vector.conjugate() @ qk_vector
    print("qk_val: ", qk_val)

    meps = 0.5*eps
    
    magic_samples = 2*math.pow((math.pow(2.,gamma*t/2) + math.sqrt(qk_val))/(math.sqrt(qk_val + meps) - math.sqrt(qk_val)), 2.)*math.log(2*math.e*math.e/(delta/2))
    equatorial_samples = math.pow((qk_val + meps)/(eps-meps), 2.)*math.log(1/(delta/2.))
    magic_samples = int(round(magic_samples))
    equatorial_samples = int(round(equatorial_samples))
    print(magic_samples)
    print(equatorial_samples)
    v1 = cPSCS.main_simulation_algorithm(qubits,magic_samples, equatorial_samples, 23423,  1, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
    print(v1)
    #v2 = cPSCS.main_simulation_algorithm2(qubits,magic_samples, equatorial_samples, 23423,  1, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
    #print(v2)
    
if __name__ == "__multi__":
    qubits = 16
    print(qubits)
    circs = 1
    depth = 5000
    t= 10
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
    
    delta = 0.001
    gamma = math.log2(4-2*math.sqrt(2))
    seed = 411111
    compute_threshold = 10
    random.seed(seed)
    print("seed: ", seed)
    threads = 1
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
        print("i:", i)
        print("qk_val: ", qk_val)
        
        for eps in epss:
            seedvec = [random.randrange(1,10000) for _ in range(threads)]
            meps = 0.5*eps

            magic_samples = 2*math.pow((math.pow(2.,gamma*t/2) + math.sqrt(qk_val))/(math.sqrt(qk_val + meps) - math.sqrt(qk_val)), 2.)*math.log(2*math.e*math.e/(delta/2))
            equatorial_samples = math.pow((qk_val + meps)/(eps-meps), 2.)*math.log(1/(delta/2.))
            magic_samples = int(round(magic_samples))
            equatorial_samples = int(round(equatorial_samples))
            best_ratio = 0.6
            for ratio in range(1,99):
                meps2 = ratio*eps/100.
                magic_samplesr = 2*math.pow((math.pow(2.,gamma*t/2) + math.sqrt(qk_val))/(math.sqrt(qk_val + meps2) - math.sqrt(qk_val)), 2.)*math.log(2*math.e*math.e/(delta/2))
                equatorial_samplesr = math.pow((qk_val + meps2)/(eps-meps2), 2.)*math.log(1/(delta/2.))
                magic_samplesr = int(round(magic_samplesr))
                equatorial_samplesr = int(round(equatorial_samplesr))

                if 0 < magic_samplesr*equatorial_samplesr < magic_samples*equatorial_samples:
                    best_ratio = ratio
                    magic_samples = magic_samplesr
                    equatorial_samples = equatorial_samplesr
            print(eps, best_ratio, magic_samples, equatorial_samples)
                    
            def run_algorithm_1(seed):
                d_time = time.monotonic()
                v1 = cPSCS.magic_sample_1(qubits, magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
                s1_time = time.monotonic() - d_time

                return (v1, s1_time)
            
            def run_main_algorithm(seed):
                d_time = time.monotonic()
                v1 = cPSCS.main_simulation_algorithm(qubits,compute_threshold,magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  8, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
                s1_time = time.monotonic() - d_time

                return (v1, s1_time)

            def run_main_algorithm2(seed):
                d_time = time.monotonic()
                v2 = cPSCS.main_simulation_algorithm2(qubits,magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  8, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
                s2_time = time.monotonic() - d_time

                return (v2, s2_time)

            

            with Pool(threads) as p:
                #ans1 = p.map(run_algorithm_1, [random.randrange(1,10000) for _ in range(threads)])
                #total_time1 = 0
                #mean_val1 = 0
                
                #for val, time_val in ans1:
                #    mean_val1 += val/threads
                #    total_time1 += time_val
                    
                #print(ans)
                #print("algorithm_1:     ", mean_val1, total_time1)
                
                ans2 = p.map(run_main_algorithm, seedvec)
                total_time2 = 0
                mean_val2 = 0
                
                for val, time_val in ans2:
                    mean_val2 += val/threads
                    total_time2 += time_val
                print("main_algorithm:  ", mean_val2, total_time2)
                
                #ans3 = p.map(run_main_algorithm2, seedvec)
                #total_time3 = 0
                #mean_val3 = 0
                
                #for val, time_val in ans3:
                #    mean_val3 += val/threads
                #    total_time3 += time_val
                #print("main_algorithm2: ", mean_val3, total_time3)
                print("-----------------------------------")

if __name__ == "__lhs_rank_test___":
    qubits = 40
    print(qubits)
    circs = 10000
    depth = 10000
    t= 50
    print("t=", t)
    measured_qubits = 10
    
    #magic_samples = 1000
    #equatorial_samples = 1000
    #seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(measured_qubits)] + [0 for _ in range(qubits-measured_qubits)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)

    #seed = 411111
    #random.seed(seed)
    data = []
    xs = []
    ys = []
    #print("seed: ", seed)
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        #print(i)
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

        val = cPSCS.lhs_rank_info(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray)
        print(i, val)
        if type(val) == type(tuple()) and len(val) == 4:
            data.append(val)
            #xs.append(val[1])
            #ys.append(val[2])
            #print(xs[-1], ys[-1])
        else:
            print("inconsistent")
        
    import numpy as np
    from matplotlib import pyplot as plt
    data = np.array(data)
    m = min(data[:,3])
    M = max(data[:,3])
    bins = np.array(range(m, M+1))
    plt.hist(data[:,3],bins=bins)
    plt.title("With new T contraints")
    plt.xlabel("rank")
    plt.ylabel("count")
    plt.show()
            
if __name__ == "__single-process__":
    # do some simulations make some graphs
    qubits = 16
    print(qubits)
    circs = 1
    depth = 5000
    t= 8
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
    epss = [0.01,0.001]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    #random.seed(15111)
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
        #d_time = time.monotonic_ns()
        sim = qk.QiskitSimulator()
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        #qk_time += time.monotonic_ns() - d_time
        
        #qk_val = 0.05145145654395905
        
        #magic_samples = int(round(((2*(math.pow(2,gamma*t/2)+qk_val)**2)/((math.sqrt( (1+relative_eps)*qk_val) - math.sqrt(qk_val))**2))/(math.log(delta/(2*relative_eps*relative_eps*qk_val*qk_val)))))
        #equatorial_samples = int(round(((((1+relative_eps))/(relative_eps))**2)*(-np.log(delta))))
        for eps in epss:
            print(eps)
            meps = 0.5*eps
            magic_samples = int(round((2*((pow(2,gamma*t/2) +qk_val)**2)/((math.sqrt(qk_val + meps) - math.sqrt(qk_val))**2))*(-math.log(delta/(2*math.e*math.e)))))
            equatorial_samples = int(round((((qk_val + meps)/(eps-meps))**2)*(-np.log(delta))))
            
            print(magic_samples)
            print(equatorial_samples)
            print("qk:", qk_val)
            #d_time = time.monotonic_ns()
            
            v1 = cPSCS.magic_sample_1(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            #s1_time += time.monotonic_ns() - d_time
            print("v1:", v1)
            #print(s1_time/1e9)
            
            #d_time = time.monotonic_ns()
            #v2 = cPSCS.magic_sample_2(qubits,magic_samples, equatorial_samples, seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
            #s2_time += time.monotonic_ns() - d_time
            
            #print("v2:", v2)

            v3 = cPSCS.main_simulation_algorithm(qubits, magic_samples, equatorial_samples, seed,  8, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
            #print("qk:",qk_val)
            print("v3: ", v3)

            
            # print(s2_time/1e9)
            
            #print()
            #print(qk_time/1e9, s1_time/1e9,s2_time/1e9)
            #print("-------------------")

