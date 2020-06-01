import numpy as np
from chstate import CHState
import constants
import gates 
import util 
from agstate import AGState
from copy import deepcopy

class MeasurementOutcome(gates.Gate):
    def __init__(self, x: np.ndarray, gates=None): 
        self.x = np.uint8(x)

    #given out outcome <v|, where v is a binary vector of length 2^n
    #and an n-qubit stabiliser state w UC UH |s> = 
    #calculate <v|s>
    def apply(self, state: CHState) -> complex:

        u = ( self.x @ state.F % 2 ) # u[j] stores the exponent of X_j (either 0 or 1)

        if any(u[state.v == 0] != state.s[state.v == 0]) or abs(state.phase) < 10e-10 : # the second product of (56) is equal to 0
            return 0
        else: # the second product of (56) is equal to 1
            signBit = util.sort_pauli_string(state.F[self.x != 0], state.M[self.x != 0] )
            signBit = signBit + (u[state.v == 1] @ state.s[state.v == 1]) % 2 #first product in (56)
            
            phase = complex(0,1)**(state.g[self.x != 0].sum() % constants.UNSIGNED_4) * ((-1)**signBit)
            return phase * np.power(2, -(state.v.sum()/2.)) * state.phase
    def __str__(self):
        return "<" + "".join(self.x) + "|"

    def applyAG(self, state: AGState) -> complex:
        for gate in self.gates:
            gate.applyAG(state)

        cpy = deepcopy(state)
        for i, val in enumerate(self.x):
            if val:
                cliffords.XGate(i).applyAG(cpy)
                
        cpy.gausStab()
        
        s = 0
        for i in range(cpy.N, 2*cpy.N):
            if cpy.r[i] == 1 and cpy.x[i].sum() == 0:
                return 0
            else:
                if  cpy.x[i].sum() > 0:
                    s += 1                
        
        return (1/np.sqrt(2))**s


