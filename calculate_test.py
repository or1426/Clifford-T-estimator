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
from multiprocessing import Pool
import time


if __name__ == "__main__":
    small_qubit_number = 10
    big_qubit_number = 53

    big_circuit_t = 10
    small_circuit_t = 20

    depth = 1000
    circs = 10

    #aArray = np.array([0 for _ in range(measured_qubits)], dtype=np.uint8)
    #print("qubits = ", qubits)
    #print("measured_qubits = ", measured_qubits)
    #print("t = ", t)

    qk_vals = np.zeros(circs)
    calc_vals = np.zeros(circs)

    for  i, (small, big) in enumerate(zip(
            util.random_clifford_circuits_with_bounded_T(small_qubit_number, depth, circs, small_circuit_t),
            util.random_clifford_circuits_with_bounded_T(big_qubit_number, depth, circs, big_circuit_t))):
        
        bigDagger  = big.inverse()
        gateArray = np.zeros(len(small.gates) + len(big.gates) + len(bigDagger.gates), dtype=np.uint8)
        controlArray = np.zeros(len(small.gates) + len(big.gates) + len(bigDagger.gates), dtype=np.uint)
        targetArray = np.zeros(len(small.gates) + len(big.gates) + len(bigDagger.gates), dtype=np.uint)
        
        measured_qubits = small_qubit_number
        aArray = np.zeros(measured_qubits, dtype=np.uint8) #np.random.randint(0,2,measured_qubits, dtype=np.uint8)
        
        for j, gate in enumerate(small.gates):
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

        for j, gate in enumerate(big.gates):
            if isinstance(gate, CXGate):
                gateArray[j+depth] = 88 #X
                controlArray[j+depth] = gate.control
                targetArray[j+depth] = gate.target
            elif isinstance(gate, CZGate):
                gateArray[j+depth] = 90 #Z
                controlArray[j+depth] = gate.control
                targetArray[j+depth] = gate.target
            elif isinstance(gate, SGate):
                gateArray[j+depth] = 115 #s
                targetArray[j+depth] = gate.target
            elif isinstance(gate, HGate):
                gateArray[j+depth] = 104 #h
                targetArray[j+depth] = gate.target
            elif isinstance(gate, TGate):
                gateArray[j+depth] = 116 # t
                targetArray[j+depth] = gate.target



        for j, gate in enumerate(bigDagger.gates):
            if isinstance(gate, CXGate):
                gateArray[j+2*depth] = 88 #X
                controlArray[j+2*depth] = gate.control
                targetArray[j+2*depth] = gate.target
            elif isinstance(gate, CZGate):
                gateArray[j+2*depth] = 90 #Z
                controlArray[j+2*depth] = gate.control
                targetArray[j+2*depth] = gate.target
            elif isinstance(gate, SGate):
                gateArray[j+2*depth] = 115 #s
                targetArray[j+2*depth] = gate.target
            elif isinstance(gate, HGate):
                gateArray[j+2*depth] = 104 #h
                targetArray[j+2*depth] = gate.target
            elif isinstance(gate, TGate):
                gateArray[j+2*depth] = 116 # t
                targetArray[j+2*depth] = gate.target

        for j, a in enumerate(aArray):
            small | PauliZProjector(target=j, a=a)

        start = time.monotonic()
        sim = qk.QiskitSimulator()
        
        qk_vector = sim.run(small_qubit_number, np.zeros(small_qubit_number), small)
        qk_val = qk_vector.conjugate() @ qk_vector
        qk_vals[i] = abs(qk_val)

        qk_time = time.monotonic() - start
        print("qk took:", qk_time)
        start = time.monotonic()
        p = cPSCS.calculate_algorithm(big_qubit_number, measured_qubits, np.copy(gateArray), np.copy(controlArray), np.copy(targetArray), np.copy(aArray))
        calc_time = time.monotonic() - start
        print("calc took", calc_time)
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
