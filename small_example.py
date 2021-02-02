#! /usr/bin/env python3
import numpy

from gates import SGate, CXGate, CZGate, HGate
from gates import TGate

import util 

import clifford_t_estim # this is the module containing out C code

import estimate # the python estimate code wraps the C implementation of RawEstim

if __name__ == "__main__":

    #first setup a simple quantum circuit
    #note our initial state is always |0>^n = [1,0,0....0]
    qubits = 2
    circ = HGate(0) | TGate(0) | HGate(0)

    #we have two qubits labelled 0 and 1, we don't do anything to one of them and the other gets H T H applied
    # H T H corresponds to a rotation about the X axis of the Bloch sphere by pi/4
    #following this rotation the probability of obtaining the outcome 0 in a computational-basis measurement of the first qubit is (1 + 1/sqrt(2))/2, roughly 0.854 
    
    # the c code is given the circuit in the form of 3 numpy arrays, for convenience this function does the transformation
    gates, controls, targets = util.convert_circuit_to_numpy_arrays(circ)

    measured_qubits = 1 #we measure only the first qubit

    measurement_outcome = numpy.array([0], dtype=numpy.uint8) # we seek the probability for measurement outcome 0

    
    d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, v, pyChState, pyAGState, magic_arr = clifford_t_estim.compress_algorithm(qubits, measured_qubits, gates, controls, targets, measurement_outcome)

    print("Compress output:")
    print("    d =", d)
    print("    r =", r)
    print("    t =", t)
    print("    v =", v)
    print("    final_t =", final_t)
    
    #we obtain
    #d = 1
    #r = 0
    #t = 1
    #v = 0
    #final_t = 1

    #indicating that Compress has not been able to remove the T gate (the final_t count = 1) this is expected as the circuit is not Clifford so must contain at least one T gate

    #now lets see what the Compute algorithm does with this circuit
    print("Analytic probability = ", (1+numpy.cos(numpy.pi/4))/2)
    gates, controls, targets = util.convert_circuit_to_numpy_arrays(circ) # note the Compress algorithm code changes the gate arrays in place to do gagetization
    prob_compute = clifford_t_estim.compute_algorithm(qubits, measured_qubits, gates, controls, targets, measurement_outcome)

    print("Compute probability =", prob_compute)
    m = numpy.sqrt(4.-2*numpy.sqrt(2))   #sqrt(stabilizer extent)
    prob_estimate, eps = estimate.estimate(epsTot=0.01, deltaTot=1e-5, t=t, measured_qubits=measured_qubits, r=r, v=v, m=m, CH=pyChState, AG=pyAGState)

    print("Estimate probability =", prob_estimate, "+/-", eps)
    
