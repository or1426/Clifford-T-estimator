from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import util
import time
#import myModule
@dataclass
class CHState:
    N : int # number of qubits
    A : np.ndarray # NxN matrix of bytes (we are using as bits) partly determines U_C
    B : np.ndarray # NxN matrix of bytes (we are using as bits) partly determines U_C
    C : np.ndarray # NxN matrix of bytes (we are using as bits) partly determines U_C
    g : np.ndarray # gamma is in (Z / Z^4)^N
    v : np.ndarray # array of N bytes (which we are using as bits) determining U_H 
    s : np.ndarray #array of N bytes (which we are using as bits) - the initial state 
    phase : complex #initial phase

    @classmethod
    def basis(cls, N:int = None, s=None) -> CHState:
        """
        Return a computational basis state defined by the bitstring s
        """
        if N == None and s is None:
            #given no input we assume a single qubit in state |0>
            return cls(N=1,
                       A=np.eye(1, dtype=np.uint8),
                       B=np.eye(1, dtype=np.uint8),
                       C=np.zeros((1,1), dtype=np.uint8),
                       g=np.zeros(1, dtype=np.uint8),
                       v=np.zeros(1, dtype=np.uint8),
                       s=np.zeros(1, dtype=np.uint8),
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
                return CHState.basis(N=None, s = s[:N])
            else:
                return CHState.basis(N=None, s = np.concatenate((s, np.zeros(N-len(s), dtype=np.uint8))))  

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
        return other.applyCH(self)

    def _rowToStr(row):
        return "".join(map(str,row))

    def tab(self):
        """
        pretty "to string" method for small qubit numbers
        prints blocks F G M gamma v s
        with headings to indicate which is which
        """
        
        s  = str(self.N) + " "
        qubitNumberStrLen = len(s)
        matrix_width = self.N
        half_matrix_width = self.N//2
        s = "N" + " "*(qubitNumberStrLen -1 + half_matrix_width) + "F" + " "*matrix_width + "G" + " "*matrix_width + "M" + " "*(matrix_width-half_matrix_width) + "g v s w\n" + s
        
        for i, (Fr, Gr, Mr, gr, vr, sr) in enumerate(zip(self.F, self.G, self.M, self.g, self.v, self.s)):
            if i != 0:
                s += " "*qubitNumberStrLen
            s += CHState._rowToStr(Fr) + " " + CHState._rowToStr(Gr) + " " + CHState._rowToStr(Mr) + " " + str(gr) + " " + str(vr) + " " + str(sr)

            if i == 0:
                s += " " + str(self.phase)
            s += "\n"
        return s

        
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
            s += CHState._rowToStr(Fr) + " " + CHState._rowToStr(Gr) + " " + CHState._rowToStr(Mr) + " " + str(gr) + " " + str(vr) + " " + str(sr)

            if i == 0:
                s += " " + str(self.phase)
            s += "\n"
        return s

    def delete_qubit(self, k):
        mask = np.ones(self.N,dtype=bool)
        mask[k] = False
        mask2d = np.outer(mask,mask)
        
        return CHState(self.N-1, self.A[mask2d].reshape(self.N-1, self.N-1), self.B[mask2d].reshape(self.N-1, self.N-1), self.C[mask2d].reshape(self.N-1, self.N-1), self.g[mask], self.v[mask],self.s[mask], self.phase)
        


    def __sub__(self,other):
        return CHState(self.N,
                         (self.A - other.A)%np.uint8(2),
                         (self.B - other.B)%np.uint8(2),
                         (self.C - other.C)%np.uint8(2),
                         (self.g - other.g)%np.uint8(4),
                         (self.v - other.v)%np.uint8(2),
                         (self.s - other.s)%np.uint8(2),
                         (self.phase / other.phase))

    def __add__(self,other):
        return CHState(self.N,
                         (self.A + other.A)%np.uint8(2),
                         (self.B + other.B)%np.uint8(2),
                         (self.C + other.C)%np.uint8(2),
                         (self.g + other.g)%np.uint8(4),
                         (self.v + other.v)%np.uint8(2),
                         (self.s + other.s)%np.uint8(2),
                         (self.phase * other.phase))

    def equatorial_inner_product(self, A):
        """
        Given an equatorial state |phi_A> defined by a symmetric binary matrix A
        compute
        <|phi_A | self >
        """
        J = np.int64((self.M @ self.F.T) % np.uint8(2))
        J[np.diag_indices_from(J)] = self.g
        K = (self.G.T @ (A + J) @ self.G)
        prefactor = (2**(-(self.N + self.v.sum())/2)) * ((1j)**(self.s @ K @ self.s)) * ((-1)**(self.s @ self.v))
        B = (K + 2*np.diag(self.s + self.s @ K))[self.v == 1][:,self.v == 1] % np.uint8(4)

        #M = np.triu(B) % np.uint8(2) #upper triangular part including diagonal
        #M[np.diag_indices_from(M)] = np.uint8(0)
        #print("M = ", M)
        K = B[np.diag_indices_from(B)] % np.uint8(2)

        L = ((B[np.diag_indices_from(B)] - K) // np.uint8(2))  # the // forces integer division and makes sure the dtype remains uint8

        newL = np.append(L,0)

        newM = np.triu(B +np.outer(K,K)) %np.uint8(2)
        newM[np.diag_indices_from(newM)] = np.uint8(0)
        
        newM = np.concatenate((newM, np.array([K],dtype=np.uint8)), axis=0)
        newM = np.concatenate((newM, np.array([[0]*newM.shape[0]],dtype=np.uint8).T) ,  axis=1)
        newM = np.uint8(newM)
        
        newL = np.uint8(newL)

        # m2 = np.copy(newM)
        # l2 = np.copy(newL)
        # m3 = np.copy(newM)
        # l3 = np.copy(newL)
        
        #re = util.slowZ2ExponentialSum(newM, newL) / 2
        #newL[-1] = 1
        #im = util.slowZ2ExponentialSum(newM, newL) / 2
        
        re  = util.z2ExponentialSum(m2, l2)
        l2[-1] = 1
        im  = util.z2ExponentialSum(m2, l2)   
        re /= 2
        im /= 2

        #delta = time.monotonic()
        #re, im = util.z2DoubleExponentialSum2(newM,newL)
        
        
        #re, im = myModule.exponential_sum(newM,newL)
        #re /= 2
        #im /= 2

        #delta = time.monotonic() - delta
        # re2, im2 = util.z2DoubleExponentialSum2(m2,l2)
        # re2 /=2
        # im2 /=2

        # re3  = util.z2ExponentialSum(m3, l3)
        # l3[-1] = 1
        # im3  = util.z2ExponentialSum(m3, l3)
        
        # re3 /= 2
        # im3 /= 2

        # good = abs(re-re2) + abs(im-im2) < 1e-6
        # if not good:
        #     print(re, im)
        #     print(re2, im2)
        #     print(re3, im3)
        #     print()

        
        return self.phase.conjugate()*prefactor*complex(re, im )

        
        
    def __eq__(self, other):
        if not isinstance(other, CHState):
            return False

        if (self.F == other.F).all() and \
           (self.G == other.G).all() and \
           (self.M == other.M).all() and \
           (self.g == other.g).all() and \
           (self.v == other.v).all() and \
           (self.s == other.s).all() and abs(self.phase - other.phase) < 1e-10:
            return True
           
            
