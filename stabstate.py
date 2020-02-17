from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class StabState():
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
