#! /usr/bin/env python3
from __future__ import annotations

import numpy as np
from measurement import MeasurementOutcome
from chstate import CHState
import constants
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector
from gates import TGate, CompositeGate
import itertools                    
import util
import random
import magicsim

import qk
from agstate import AGState
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit import Aer

def basisTuple(n, *args):
    v = [0]*n
    for arg in args:
        v[arg] = 1
    return tuple(v)

from numpy import linalg

def eq(a,b,eps=1e-10):
    return linalg.norm(a - b) < eps
    

from numpy import linalg

if __name__ == "__XY-test__":
    qubits = 1
    state = CHState.basis(qubits)
    state | HGate(0)

    print(state.tab())
    
    equatorial_state = np.array([[1]], dtype=np.uint8)
    p1 = state.equatorial_inner_product(equatorial_state)
    
    p2 = 0
    for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
        #reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
        x = np.array(t)
        p2 +=  MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state).conjugate() * ((1j)**(np.int(x @ equatorial_state @ x)))
        
    p2 = p2 / (2**(qubits/2))
    
    print(p1, p2)
    print(abs(p1-p2) < 1e-10)

if __name__ == "__main__":
    qubits = 8
    tests = 1000
    depth = 100

    for i, circ in enumerate(util.random_clifford_circuits(qubits, depth, tests)):
        print(i)
        state = CHState.basis(qubits)
        #state.s = np.randint(0, 2, qubits, dtype=np.uint8)
        #state.v = np.randint(0, 2, qubits, dtype=np.uint8)

        state = state | circ

        
        
        equatorial_state = np.random.randint(0, 2, (qubits,qubits), dtype=np.uint8)
        equatorial_state = ((equatorial_state+equatorial_state.T)//2) % np.uint8(2) # have to be symmetric
        p1 = state.equatorial_inner_product(equatorial_state)

        p2 = 0
        for i, t in enumerate(itertools.product([0,1], repeat=qubits)):
            #reconstructed_vector[i] = MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state)
            x = np.array(t)
            p2 +=  MeasurementOutcome(np.array(t, dtype=np.uint8)).apply(state).conjugate() * ((1j)**(np.int(x @ equatorial_state @ x)))

        p2 = p2 / (2**(qubits/2))

        print(p1, p2)
        print(abs(p1-p2) < 1e-10)
        if abs(p1-p2) > 1e-10:
            break
        
        
    
if __name__ == "__exponential-sum-test__":
    qubits = 10
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



        
    
