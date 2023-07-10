from dataclasses import dataclass
from gates import CompositeGate, CompositeCliffordGate, TGate, CXGate, CZGate, HGate, SGate
import numpy as np

@dataclass
class GadgetizedStabState:
    n : int # number of qubits
    k : int # number of stabilizers 
    x : np.ndarray
    z : np.ndarray
    r : np.ndarray
    
    @classmethod
    def __init__(self, n, circ):
        self.t = 0
        _circ = circ
        if isinstance(circ, CompositeGate) or isinstance(circ, CompositeCliffordGate):
            _circ = circ.gates
            
        for gate in _circ:
            if isinstance(gate, TGate):
                self.t += 1                
        self.n = n
        self.k = n
        self.x = np.zeros((self.k + self.t, self.n + self.t), dtype=np.uint8)
        self.z = np.eye(self.k + self.t, self.n + self.t, dtype=np.uint8)
        self.r = np.zeros(self.k + self.t, dtype=np.uint8)
        
        t_so_far = 0
        
        for gate in _circ:
            if isinstance(gate, TGate):
                GadgetizedStabState.CX(self, gate.target, t_so_far)
                t_so_far += 1
            elif isinstance(gate, CZGate):
                GadgetizedStabState.CZ(self, gate.control, gate.target)
            elif isinstance(gate, CXGate):
                GadgetizedStabState.CX(self, gate.control, gate.target)
            elif isinstance(gate, HGate):
                GadgetizedStabState.H(self, gate.target)
            elif isinstance(gate, SGate):
                GadgetizedStabState.S(self, gate.target)
                

    def __str__(self):
        stabilizer_strings = []
                
        for k, _ in enumerate(self.x):
            s = []
            if k == self.k:
                s.append('\n')            
            if self.r[k]:
                s.append("-")
            else:
                s.append("+")
            for i, (x, z) in enumerate(zip(self.x[k], self.z[k])):
                if i == self.n:
                    s.append(" ")
                if x==0 and z==0:
                    s.append("I")
                if x==1 and z==0:
                    s.append("X")
                if x==0 and z==1:
                    s.append("Z")
                if x==1 and z==1:
                    s.append("Y")
            stabilizer_strings.append("".join(s))
        return '\n'.join(stabilizer_strings)
            
    def _g(x1,z1,x2,z2):
        return x1*z1*(z2 - np.int8(x2)) + x1*((z1 + 1) % np.uint8(2))*z2*(2*np.int8(x2) -1) + ((x1 + 1) % np.uint8(2)) *z1*x2*(1-2*np.int8(z2))

    def _g2(x1,z1,x2,z2):
        if x1 == 1 and z1 == 1:
            return z2 - np.int8(x2)
        elif x1 == 1 and z1 == 0:
            return z2*(2*x2-1)
        elif x1 == 0 and z1 == 1:
            return x2*(1-2*z2)
        else:
            return 0
        
    def rowsum(self, h, i):
        s = 2*self.r[h] + 2*self.r[i] + AGState._g(self.x[i], self.z[i], self.x[h], self.z[h]).sum()
        s = s % np.uint8(4)        
        
        self.r[h] = s//np.uint8(2)
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]
            
    def rowswap(self, h, i):
        self.x[[h,i]] = self.x[[i,h]]
        self.z[[h,i]] = self.z[[i,h]]
        self.r[[h,i]] = self.r[[i,h]]

    def H(self, q):
        for i in range(self.k + self.t):
            self.r[i] ^= self.x[i][q] * self.z[i][q] 
            self.x[i][q], self.z[i][q] = self.z[i][q], self.x[i][q]
    def S(self, q):
        for i in range(self.k + self.t):
            self.r[i] ^= self.x[i][q] * self.z[i][q]
            self.z[i][q] ^= self.x[i][q]

    def CX(self, a, b):
        for i in range(self.k + self.t):
            self.r[i] ^= self.x[i][a] * self.z[i][b] * (self.x[i][b] ^ self.z[i][a] ^ 1)
            self.x[i][b] ^= self.x[i][a]
            self.z[i][a] ^= self.z[i][b]
            
    def CZ(self, a,b):
        GadgetizedStabState.H(self, b)
        GadgetizedStabState.CX(self, a, b)
        GadgetizedStabState.H(self, b)

        
    def phaseless_cb_inner_product(self):
        """
        here we do something a bit strange
        we compute the biggest inner product between our state an a computational basis state
        """
        rank = 0
        for col in range(self.n):
            pivot = -1
            for row in range(rank, self.k):
                if self.x[row][col] == 1:
                    pivot = row
                    break
            if pivot >= 0:
                self.rowswap(rank, pivot)
                for i in range(0, self.k):
                    if i != rank and self.x[i][col] == 1:
                        self.rowsum(i,rank)
                rank += 1

        #so now we have stuff above this line with xs in and stuff below with no xs
        #to keep the rows linearly independent we therefore have the remaining rows forming
        #a linearly indep set of Z guys
        return 2**(-((self.n-self.k) + rank))
