import numpy as np
from stabstate import StabState
import constants

class MeasurementOutcome(object):
    def __init__(self, x: np.ndarray):
        self.x = np.bool_(x)

    #given out outcome <v|, where v is a binary vector of length 2^n
    #and an n-qubit stabiliser state w UC UH |s> = 
    #calculate <v|s>
    def overlap(self, state: StabState) -> complex: 
        
        u = ( self.x @ state.A % 2 )
        if any(u[state.v == 0] != state.s[state.v == 0]): # the second product of (56) is equal to 0
            return 0
        else: # the second product of (56) is equal to 1
            signBit = False # False == 0, True == 1, we want to keep track of powers of (-1) from commuting Z and X operators
            
            t = np.zeros_like(state.v) # t[j] stores the exponent of Z_j (either 0 or 1)
            for p in np.flatnonzero(self.x): # we put a Pauli X on each qubit where the bitstring self.x is 1
                # we multiply by UC^{-1} X_p UC = i^{gamma_p} prod_ X_j^{F_{pj}} Z_j^{M_{pj}}
                # first we need to commute each X_j^{F_{pj}} through Z_j^t[j]
                #this gives a (-1) for each j for which F_{pj} and t[j] are both non-zero

                signBit = signBit ^ (t @ state.A[p] % 2)

                t = t ^ state.C[p]

            
            signBit = (signBit + u @ t) % 2
            signBit = signBit + (u[state.v == 1] @ state.s[state.v == 1]) %2
            phase = complex(0,1)**(state.g[self.x != 0].sum() % constants.UNSIGNED_4) * ((-1)**signBit)
            return phase * np.power(2, -(state.v.sum()/2.))
