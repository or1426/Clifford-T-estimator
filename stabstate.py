from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class StabState:
    N : int # number of qubits
    A : np.ndarray # NxN matrix of bytes (we are using as bits) partly determines U_C
    B : np.ndarray # NxN matrix of bytes (we are using as bits) partly determines U_C
    C : np.ndarray # NxN matrix of bytes (we are using as bits) partly determines U_C
    g : np.ndarray # gamma is in (Z / Z^4)^N
    v : np.ndarray # array of N bytes (which we are using as bits) determining U_H 
    s : np.ndarray #array of N bytes (which we are using as bits) - the initial state 
    phase : complex #initial phase

    @classmethod
    def basis(cls, N:int = None, s=None) -> StabState:
        """
        Return a computational basis state defined by the bitstring s
        """
        if N == None and s is None:
            #given no input we assume a single qubit in state |0>
            return cls(N=2,
                       A=np.eye(2, dtype=np.uint8),
                       B=np.eye(2, dtype=np.uint8),
                       C=np.zeros((2,2), dtype=np.uint8),
                       g=np.zeros(N, dtype=np.uint8),
                       v=np.zeros(2, dtype=np.uint8),
                       s=np.zeros(2, dtype=np.uint8),
                       phase = complex(1,0)
            )
        elif N != None and s is None:
            #we get given a number of qubits but no other information so return the state |0,0....0>
            return cls(N=N,
                       A=np.eye(N, dtype=np.uint8),
                       B=np.eye(N, dtype=np.uint8),
                       C=np.zeros((N,N), dtype=np.uint8),
                       g=np.zeros(N, dtype=np.uint8),
                       v=np.zeros(N, dtype=np.uint8),
                       s=np.zeros(N, dtype=np.uint8),
                       phase = complex(1,0)
            )
        elif N == None and not s is None:
            #we get given some bitstring so we return that computational basis state
            N = len(s)
            s = np.array(s, dtype=np.uint8) #we accept lists etc, but convert them to np arrays
            return cls(N=N,
                       A=np.eye(N, dtype=np.uint8),
                       B=np.eye(N, dtype=np.uint8),
                       C=np.zeros((N,N), dtype=np.uint8),
                       g=np.zeros(N, dtype=np.uint8),
                       v=np.zeros(N, dtype=np.uint8),
                       s=s,
                       phase = complex(1,0)
            )
        else:
            #both N and s are not none
            # if N <= len(s) we truncate s to length s and proceed as before
            # if N > len(s) we extend s by adding zeros at the end and proceed as before
            s = np.array(s, dtype=np.uint8)
            if N <= len(s):
                return StabState.basis(N=None, s = s[:N])
            else:
                return StabState.basis(N=None, s = np.concatenate((s, np.zeros(N-len(s), dtype=np.uint8))))  

    @property
    def F(self):
        return self.A
    @F.setter
    def F(self, mat):
        self.A = mat
    @property
    def G(self):
        return self.B
    @G.setter
    def G(self, mat):
        self.B = mat
    @property
    def M(self):
        return self.C
    @M.setter
    def M(self, mat):
        self.C = mat
    @property
    def gamma(self):
        return self.g
    @gamma.setter
    def gamma(self, mat):
        self.g = mat
    @property
    def w(self):
        return self.phase
    @w.setter
    def w(self, c):
        self.phase = c

    def __or__(self, other : CliffordGate):
        return other.apply(self)

    def _rowToStr(row):
        return "".join(map(str,row))
    
    def __str__(self):
        """
        pretty "to string" method for small qubit numbers
        prints blocks F G M gamma v s
        """
        qubitNumberStrLen = None
        s = ""
        
        for i, (Fr, Gr, Mr, gr, vr, sr) in enumerate(zip(self.F, self.G, self.M, self.g, self.v, self.s)):
            if i == 0:
                s = str(self.N) + " "
                qubitNumberStrLen = len(s)
            if i != 0:
                s += " "*qubitNumberStrLen
            s += StabState._rowToStr(Fr) + " " + StabState._rowToStr(Gr) + " " + StabState._rowToStr(Mr) + " " + str(gr) + " " + str(vr) + " " + str(sr)

            if i == 0:
                s += " " + str(self.phase)
            s += "\n"
        return s

    def delete_qubit(self, k):
        mask = np.ones(self.N,dtype=bool)
        mask[k] = False
        mask2d = np.outer(mask,mask)
        
        return StabState(self.N-1, self.A[mask2d].reshape(self.N-1, self.N-1), self.B[mask2d].reshape(self.N-1, self.N-1), self.C[mask2d].reshape(self.N-1, self.N-1), self.g[mask], self.v[mask],self.s[mask], self.phase)
        
