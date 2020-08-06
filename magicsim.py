from gates import CompositeCliffordGate, TGate, HGate, CXGate, SGate
from chstate import CHState
from agstate import AGState
import util
from copy import deepcopy
import numpy as np

class TrivialSimulator(object):
    def __init__(self, n, circ = None):
        self.n = n
        self.circ = []
        self.t = 0

        for gate in circ.gates:
            if isinstance(gate, TGate):
                target = gate.target
                #replace the magic gate with a cnot to the magic qubit
                #we will insert H gates on each magic qubit later
                self.circ.append(CXGate(control=target, target=self.n+self.t))
                self.t += 1
            else:
                self.circ.append(gate)

        self.circ = CompositeCliffordGate(self.circ)


    def magic_sample(self, ys):
        circ = deepcopy(self.circ)
        circ = CompositeCliffordGate([HGate(self.n+i) for i in range(self.t)]) | CompositeCliffordGate([SGate(self.n+i) for i in range(self.t) if ys[i] == 1]) | circ
        
        state = CHState.basis(self.n+self.t)

        return state | circ

        

class MagicSimulator(object):
    """
    A class to wrap up the simulator of a Clifford+T circuit
    The circuit should consist of entirely Clifford and T gates
    The initial state is assumed to be |0>^n
    The n should only include code qubits - the simulator will add magic qubits as it needs
    """
    def __init__(self, n, circ):
        
        self.circ = []

        self.n = n
        self.t = 0

        for i, gate in enumerate(circ.gates):
            if isinstance(gate, TGate):
                target = gate.target
                #replace the magic gate with a cnot to the magic qubit
                #we will insert H gates on each magic qubit later
                self.circ.append(CXGate(control=target, target=self.n+self.t))
                self.t += 1
            else:
                self.circ.append(gate)
        
        self.circ = CompositeCliffordGate([HGate(i) for i in range(self.n,self.n+self.t)]+ self.circ)
        

        #we store both CH-form and Aaronson-Gottesman states
        
        self.chState = CHState.basis(self.n+self.t)
        self.agState = AGState.basis(self.n+self.t)

        self.chState= self.circ.applyCH(self.chState)
        self.agState = self.circ.applyAG(self.agState)



    def magic_sample(self, ys):
        """
        y is a list of length t indicating which of the magic gates we put an S gate on
        if y[i] == 1 then the ith magic qubit gets an S
        if y[i] == 0 then the ith magic qubit does not get an S
        """

        #for each non-zero element in y
        #we want to multiply the initial state by HGate(i) SGate(i) HGate(i)
        #this turns out to be equivalent to multiplying the whole final state by
        #U H_k S_k H_k U^\dagger
        #but H_k S_k H_k = e^{i\pi/4} \frac{1}{\sqrt{2}} (I -i X_k)
        #so now we evolve identity forward by U (trivial)
        #and evolve X_k forward by U (using the AGState)
        #then we have to send the resulting Pauli through UC and UH
        #giving a third Pauli
        #then the state is of the form (we^{i\pi/4}) UC UH (I  + i^d P)/sqrt(2) |s>
        #then we apply Bravyi et al's prop. 4 to turn this into a new ch form
        

        chCopy = deepcopy(self.chState) #we update this copy as we go

        for i, y in enumerate(ys):
            if y:
                #we want to know what U_c^\dagger U X_i U^\dagger U_c is
                #firstly we use the A-G info
                # U X_i U^\dagger is the i'th destabiliser
                x = self.agState.x[self.n+i]
                z = self.agState.z[self.n+i]
                r = self.agState.r[self.n+i]

                #print(x,z,r)
                x_col = np.array([x]).T
                z_col = np.array([z]).T
                
                #now we apply U_c to this using the CH-form info
                x_mat = chCopy.F * x_col
                z_mat = (chCopy.M * x_col + chCopy.G*z_col) % np.uint8(2)
                r = (r + util.sort_pauli_string(x_mat, z_mat)) % np.uint8(2)

                u = (x @ chCopy.F) % np.uint8(2)
                h = (x @ chCopy.M + z @ chCopy.G) % np.uint8(2)

                g = (x @ (z + chCopy.g)) % np.uint8(4)

                #now U_c^dag U X_i U^dag U_C = (-1)^r i^g prod_j Z_j^{h_j} X_j^{u_j}
                #we want to conjugate this by U_H
                #everywhere chCopy.v == 1 we flip a z to an x and an x to a z
                #everywhere chCopy.v == 1 and u == 1 and h == 1 we need to swap the order of our x and z so we get a minus sign

                u2 = u*(np.uint8(1) ^ chCopy.v) ^ (h*chCopy.v)
                h2 =  (u*chCopy.v) ^ (h*(np.uint8(1) ^ chCopy.v))

                r = (r + (u*h*chCopy.v).sum()) % np.uint8(2)
                
                
                #now U_H^dag U_c^dag U X_i U^dag U_C U_H = (-1)^r i^g prod_j Z_j^{h2_j} X_j^{u2_j}

                t =  u2 ^ chCopy.s
                r = (r + h2 @ t) % np.uint8(2)

                #now we have w UC UH |s> = w (-1)^r (i)^g UC UH |t>

                if all(t == chCopy.s):
                    chCopy.w *= np.exp(1j*np.pi/4) * (1 + (1j)**(g+2*r -1) )/ np.sqrt(2)
                else:
                    phase, VCList, v, s = util.desuperpositionise(chCopy.s, t, (g+2*r -1)%np.uint8(4), chCopy.v)

                    chCopy.w *= phase*np.exp(1j*np.pi/4)/np.sqrt(2)
                    chCopy.v = v
                    chCopy.s = s

                    for gate in VCList:
                        gate.rightMultiplyC(chCopy)
                    
        return chCopy
                
