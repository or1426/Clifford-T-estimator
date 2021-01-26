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
#from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
import sys
#import qiskit
from matplotlib import pyplot as plt
import cPSCS
import estimate
import estimate2
from multiprocessing import Pool


#from matplotlib import pyplot as plt
#from multiprocessing import Pool

import time

def basisTuple(n, *args):
    v = [0]*n
    for arg in args:
        v[arg] = 1
    return tuple(v)

def eq(a,b,eps=1e-10):
    return linalg.norm(a - b) < eps
    

from numpy import linalg


if __name__ == "__sf__":
    qubitss = list(range(25,51))
    
    t = 80
    depth = 10000
    circs = 100
    measured_qubits = 5

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
    times = []
    #for t in ts:
    time_acc = 0
    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        gateArray = np.zeros(depth, dtype=np.uint8)
        controlArray = np.zeros(depth, dtype=np.uint)
        targetArray = np.zeros(depth, dtype=np.uint)
        #qkcirc = qiskit.QuantumCircuit(qubits)
        for j, gate in enumerate(circ.gates):
            if isinstance(gate, CXGate):
                gateArray[j] = 88 #X
                controlArray[j] = gate.control
                targetArray[j] = gate.target
                #qkcirc.cx(target_qubit=gate.target, control_qubit=gate.control)
            elif isinstance(gate, CZGate):
                gateArray[j] = 90 #Z
                controlArray[j] = gate.control
                targetArray[j] = gate.target
                #qkcirc.cx(target_qubit=gate.target, control_qubit=gate.control)
            elif isinstance(gate, SGate):
                gateArray[j] = 115 #s
                targetArray[j] = gate.target
                #qkcirc.s(gate.target)
            elif isinstance(gate, HGate):
                gateArray[j] = 104 #h
                targetArray[j] = gate.target
                #qkcirc.h(gate.target)
            elif isinstance(gate, TGate):
                gateArray[j] = 116 # t
                targetArray[j] = gate.target
                #qkcirc.t(gate.target)
                    
        #for i  in range(measured_qubits):
        #    circ | PauliZProjector(target=i, a=aArray[i])
        
        #print("starting qk sim")
        #print("1")
        #sim = qk.QiskitSimulator()
        #print("2")
        #d_time = time.monotonic()
        #qk_vector = sim.run(qubits, np.zeros(qubits,dtype=np.uint8), circ)
        #qk_time = time.monotonic - d_time
        #job = qiskit.execute(qkcirc, backend=backend, backend_options={"max_parallel_threads":6})
        #qk_vector = job.result().get_statevector(qkcirc)
        
        #qk_val = qk_vector.conjugate() @ qk_vector
        
        #qk_times.append(qk_time)
        #print("ended qk sim")
        #print("starting CALCULATE sim")
        #print("3")
        d_time = time.monotonic()
        comp_val, FE_time, calc_time = cPSCS.calculate_algorithm(qubits, measured_qubits,gateArray, controlArray, targetArray, aArray)
        
        comp_time = time.monotonic() - d_time
        time_acc += comp_time / circs
        #print("4")
        #print("ended CALCULATE sim")
        with open("qk_calculate_comparison.txt", "a") as f:
            print(qubits, qk_val, comp_val, qk_time, comp_time)
        print(qubits, t, comp_val, comp_time)
        with open("t=80-sim.txt", "w") as f:
            print(qubits, t, comp_val, comp_time, FE_time, calc_time,file = f)
            
            
        
if __name__ == "__plot__":
    qubits = 55
    depth = 100000
    circs = 100
    measured_qubits = 5

    ts, times = None, None
    ts2, times2 = None, None
    with open("hakop-t4.pkl", "rb") as f:
        ts, times = pickle.load(f)
    with open("hakop-t5.pkl", "rb") as f:
        ts2, times2 = pickle.load(f)
    ts = np.concatenate((np.array(ts),np.array(ts2)))
    times = np.concatenate((np.array(times),np.array(times2)))
    
    divided_times = times/np.power(2., ts)
    coeffs = np.polyfit(ts, divided_times, 2)
    xs = np.linspace(65, 85, 1000)
    plt.plot(xs, np.polyval(coeffs, xs), label="fit")
    #plt.plot(xs, coeffs[0], label="fit")
    plt.plot(ts, divided_times, label="data")
    plt.grid()
    plt.yscale("log")
    plt.title("n={}, w = {}, depth = {}".format(qubits, measured_qubits, depth))
    plt.xlabel("t")
    plt.ylabel("average time seconds per circ (over 100 circs)")
    plt.legend()
    plt.show()
    
if __name__ == "__calculate-test__":
    qubitss = range(26,51)
    t = 30
    depth = 1000
    circs = 100
    measured_qubits = 10

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    for qubits in qubitss:
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
                    
            #for i  in range(measured_qubits):
            #    circ | PauliZProjector(target=i, a=aArray[i])
                    
            #sim = qk.QiskitSimulator()
            
            #d_time = time.monotonic()
            #qk_vector = sim.run(qubits, np.zeros(qubits,dtype=np.uint8), circ)
            #qk_val = qk_vector.conjugate() @ qk_vector
            #qk_time = time.monotonic() - d_time
            #print(qk_time)
            d_time = time.monotonic()
            comp_val1, comp_val2, FE_time, calc_time = cPSCS.calculate_algorithm(qubits, measured_qubits,gateArray, controlArray, targetArray, aArray)
            comp_time = time.monotonic() - d_time
            
            with open("no_qk_calculate_comparison2_{}.txt".format(qubits), "a") as f:
                print(qubits, comp_val1, comp_val2, comp_time, FE_time, calc_time, file=f)
                #    #print(qubits, qk_val, comp_val, qk_time, comp_time, file=f)
            
            

if __name__ == "__r_hist__":
    qubits = 100
    t = int(sys.argv[1])
    depth = 100000
    circs = 100
    measured_qubits = 20

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)
    tMinusRs = []
    vs = []
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
        
        out = cPSCS.compress_algorithm(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray)
        if type(out) == tuple:
            tPrime, r, log_v, CH, AG = out
            tMinusRs.append(tPrime-r)
            vs.append(log_v)
        #print(t)
        #print(r)
        #print(log_v)
        #print(CH)
        #print(AG)
    hist_dict = {}
    for v in vs:
        if v in hist_dict:
            hist_dict[v] += 1
        else:
            hist_dict[v] = 1
    print("t = {}".format(t))
    #for tMr in tMinusRs:
    #    if tMr in hist_dict:
    #        hist_dict[tMr] += 1
    #    else:
    #        hist_dict[tMr] = 1
    #print("t = {}".format(t))
    for key in hist_dict.keys():
        print(key, hist_dict[key])
              
    #plt.hist(tMinusRs)
    #plt.xlabel("t-r")
    #plt.title("t = {}".format(t))
    #plt.grid()
    #plt.show()
        
    
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
    qubits = 40
    print(qubits)
    circs = 1
    depth = 5000
    t= 55
    print("t=", t)
    measured_qubits = 1
    #magic_samples = 1000
    #equatorial_samples = 1000
    #seed = 5952 #random seed we give to the c code to generate random equatorial and magic samples
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(measured_qubits)] + [0 for _ in range(qubits-measured_qubits)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    epss = [pow(0.5,1+measured_qubits)]
    #epss = [0.0001]
    
    delta = 0.001
    gamma = math.log2(4-2*math.sqrt(2))
    seed = 4111
    compute_threshold = 10
    random.seed(seed)
    print("seed: ", seed)
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
        #d_time = time.monotonic_ns()
        #sim = qk.QiskitSimulator()
        #qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        #qk_val = qk_vector.conjugate() @ qk_vector
        #qk_time += time.monotonic_ns() - d_time
        print("i:", i)
        
        qk_val = pow(0.5,measured_qubits)
        print("made up qk_val: ", qk_val)
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
            #magic_samples = 10000
            #equatorial_samples = 100
            print(eps, best_ratio, magic_samples, equatorial_samples)
                    
            def run_algorithm_1(seed):
                d_time = time.monotonic()
                v1 = cPSCS.magic_sample_1(qubits, magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
                s1_time = time.monotonic() - d_time

                return (v1, s1_time)
            
            def run_main_algorithm(seed):
                d_time = time.monotonic()
                v1 = cPSCS.main_simulation_algorithm(qubits,compute_threshold,magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
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



if __name__ == "__test__":
    qubits = 50
    print(qubits)
    circs = 1
    depth = 5000
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
    qk_time = 0
    s1_time = 0
    s2_time = 0

    epss = [pow(0.5,1+measured_qubits)]
    #epss = [0.0001]
    
    delta = 0.001
    gamma = math.log2(4-2*math.sqrt(2))
    seed = 4111
    compute_threshold = 10
    random.seed(seed)
    print("seed: ", seed)
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
        #d_time = time.monotonic_ns()
        #sim = qk.QiskitSimulator()
        #qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        #qk_val = qk_vector.conjugate() @ qk_vector
        #qk_time += time.monotonic_ns() - d_time
        print("i:", i)
        
        qk_val = pow(0.5,measured_qubits)
        print("made up qk_val: ", qk_val)
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
            #magic_samples = 10000
            #equatorial_samples = 100
            print(eps, best_ratio, magic_samples, equatorial_samples)
                    
            def run_algorithm_1(seed):
                d_time = time.monotonic()
                v1 = cPSCS.magic_sample_1(qubits, magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray), np.copy(mask))
                s1_time = time.monotonic() - d_time

                return (v1, s1_time)
            
            def run_main_algorithm(seed):
                d_time = time.monotonic()
                v1 = cPSCS.main_simulation_algorithm(qubits,compute_threshold,magic_samples, int(math.ceil(equatorial_samples/threads)), seed,  measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
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
            
if __name__ == "__main__":
    # do some simulations make some graphs
    qubits = 40
    measured_qubits = 3
    circs = 1
    depth = 10000
    t = 50
    print("n = ", qubits)
    print("w = ", measured_qubits)    
    print("t =", t)

    #magic_samples = 1000
    #equatorial_samples = 1000
    # make the w qubits the first 8
    mask = np.array([1 for _ in range(measured_qubits)] + [0 for _ in range(qubits-measured_qubits)], dtype=np.uint8)
    #project them all on to the 0 state since it doesn't really matter
    aArray = np.zeros_like(mask)
    qk_time = 0
    s1_time = 0
    s2_time = 0

    #epss = [0.1, 0.01,0.001]
    epss = [.02]
    
    delta = 0.01
    gamma = math.log2(4-2*math.sqrt(2))
    #random.seed(15111)
    #seed = random.randint(100,100000)
    seed = 81714 #6666 #9123 #34234
    #seed = 21517
    print("seed=",seed)
    random.seed(seed)
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

        #for i, (a,m) in enumerate(zip(aArray, mask)):
        #   if m:
        #       circ | PauliZProjector(target=i, a=a)

        d_time = time.monotonic()
        comp_val = cPSCS.calculate_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))        
        comp_time = time.monotonic() - d_time
        print("compute_algorithm: {:.16f}".format(comp_val), "took", comp_time, "seconds")
        
        for eps in epss:
            
            d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, CH, AG, magic_arr = cPSCS.compress_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
            print(t,r,log_v, final_t)

            beta = np.log2(4. - 2.*np.sqrt(2.));
            m = np.power(2, beta*final_t/2)
            p = estimate2.optimize(eps, delta, t, measured_qubits, r, log_v, m, CH, AG)
            print(p)



            
if __name__ == "__test__":
    qubits = 15
    t = 20
    depth = 1000
    circs = 1000    

    #aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)

    qk_vals = np.zeros(circs)
    calc_vals = np.zeros(circs)

    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        gateArray = np.zeros(len(circ.gates), dtype=np.uint8)
        controlArray = np.zeros(len(circ.gates), dtype=np.uint)
        targetArray = np.zeros(len(circ.gates), dtype=np.uint)
        measured_qubits = np.random.randint(0, qubits)
        aArray = np.zeros(measured_qubits, dtype=np.uint8) #np.random.randint(0,2,measured_qubits, dtype=np.uint8)
        
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
        for j, a in enumerate(aArray):
            circ | PauliZProjector(target=j, a=a)
        sim = qk.QiskitSimulator()
        
        qk_vector = sim.run(qubits, np.zeros(qubits), circ)
        qk_val = qk_vector.conjugate() @ qk_vector
        qk_vals[i] = abs(qk_val)

        p = cPSCS.calculate_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        
        calc_vals[i] = p
        print(i, qk_vals[i], calc_vals[i])
        if abs(qk_vals[i] - calc_vals[i]) > 1e-5:
            print(circ)
            break

    plt.plot(qk_vals, calc_vals,".")
    xs = np.linspace(0,1,100)
    plt.plot(xs,xs, '-')
    plt.grid()
    plt.show()
            

if __name__ == "__inverse-test__":
    qubits = 100
    t = 4
    depth = 5
    circs = 1
    measured_qubits = 2

    aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)
    

    for  i, circ in enumerate(util.random_clifford_circuits_with_bounded_T(qubits, depth, circs, t)):
        uDagger  = circ.inverse()
        weirdIdentity = CompositeCliffordGate()
        weirdIdentity.gates = circ.gates + uDagger.gates
        #weirdIdentity.gates = [CZGate(2, 0), TGate(0), HGate(2), TGate(1), TGate(2), SGate(2), SGate(2), SGate(2), TGate(2), SGate(1), SGate(1), SGate(1), TGate(1), HGate(2), SGate(0), SGate(0), SGate(0), TGate(0), CZGate(2, 0)
        #weirdIdentity.gates = [CXGate(0, 1), TGate(1), HGate(1), CXGate(3, 1), HGate(1), TGate(3), TGate(0), TGate(1), HGate(3), SGate(3)]
        #weirdIdentity.gates = [HGate(1), TGate(1), CXGate(1, 0), TGate(1), CXGate(1, 0)]
        #weirdIdentity.gates = [TGate(4), TGate(9), HGate(5), TGate(7), TGate(2), TGate(0), HGate(5), CXGate(5, 0), HGate(4), CXGate(8, 5), TGate(4), CXGate(6, 0), TGate(9), TGate(6), CZGate(4, 3), TGate(2), SGate(5), HGate(5), TGate(3), CXGate(3, 9), CXGate(3, 9), SGate(3), TGate(3), HGate(5), SGate(5), SGate(5), SGate(5), SGate(2), TGate(2), CZGate(4, 3), SGate(6), TGate(6), SGate(9), TGate(9), CXGate(6, 0), SGate(4), TGate(4), CXGate(8, 5), HGate(4), CXGate(5, 0), HGate(5), SGate(0), TGate(0), SGate(2), TGate(2), SGate(7), TGate(7), HGate(5), SGate(9), TGate(9), SGate(4), TGate(4)]
        #weirdIdentity = CompositeCliffordGate()

        print(", ".join([gate.__str__() for gate in weirdIdentity.gates]))
        
        gateArray = np.zeros(len(weirdIdentity.gates), dtype=np.uint8)
        controlArray = np.zeros(len(weirdIdentity.gates), dtype=np.uint)
        targetArray = np.zeros(len(weirdIdentity.gates), dtype=np.uint)
        
        for j, gate in enumerate(weirdIdentity.gates):
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
        for i, a in enumerate(aArray):
            weirdIdentity | PauliZProjector(target=i, a=a)
        #sim = qk.QiskitSimulator()
        #
        #qk_vector = sim.run(qubits, np.zeros(qubits), weirdIdentity)
        #qk_val = qk_vector.conjugate() @ qk_vector
        #print("qk_val: ", qk_val)
        
        data = cPSCS.compress_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        #if type(data) == tuple:
        #    tPrime, r, log_v, CH, AG = data
        #    print(t, tPrime, r, log_v)
        #else:
        #    print(data)

        p = cPSCS.calculate_algorithm(qubits, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        print(p)
            
