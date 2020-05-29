import numpy as np
from chstate import CHState
import constants
import cliffords
import util 
from agstate import AGState
from copy import deepcopy

class MeasurementOutcome(cliffords.CliffordGate):
    def __init__(self, x: np.ndarray, gates=None): # if gates is not None its a list of Clifford unitaries we apply to the state before computing the overlap 
        self.x = np.uint8(x)
        if gates == None:
            self.gates = []

    #given out outcome <v|, where v is a binary vector of length 2^n
    #and an n-qubit stabiliser state w UC UH |s> = 
    #calculate <v|s>
    def apply(self, state: CHState) -> complex:
        for gate in self.gates:
            gate.apply(state)

        u = ( self.x @ state.A % 2 ) # u[j] stores the exponent of X_j (either 0 or 1)

        if any(u[state.v == 0] != state.s[state.v == 0]) or abs(state.phase) < 10e-10 : # the second product of (56) is equal to 0
            return 0
        else: # the second product of (56) is equal to 1
            signBit = False # False == 0, True == 1, we want to keep track of powers of (-1) from commuting Z and X operators
            t = np.zeros_like(state.v) # t[j] stores the exponent of Z_j (either 0 or 1)
            for p in np.flatnonzero(self.x): # we put a Pauli X on each qubit where the bitstring self.x is 1
                # we multiply by UC^{-1} X_p UC = i^{gamma_p} prod_ X_j^{F_{pj}} Z_j^{M_{pj}}
                # first we need to commute each X_j^{F_{pj}} through Z_j^t[j]
                #this gives a (-1) for each j for which F_{pj} and t[j] are both non-zero
                #combine adjacent Zs
                t = t ^ state.C[p]

                #commute xs through zs
                signBit = signBit ^ (t @ state.A[p] % 2)

                #combine adjacent xs, uncecessary since there is a nice closed form for u (above)
                
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


class PauliZProjector(cliffords.CliffordGate):
    """
    Class to model a projector of the form 
    (1/2) * (I + (-1)^a Z_target)
    where a is an integer and Z_q is the unitiary applying Pauli Z to qubit q and identity to all other qubits
    """
    def __init__(self, target:int, a:int):
        self.target = target
        self.a = a 

    """
    Applying a projector to a state results in a "state" which is not normalised
    the normalisation factor will be kept in the phase of the stabaliser state
    """
    def apply(self, state: CHState) -> CHState:
        #apply commutation rules to get (1/2) (1+ (-1)^a Z_p) UC UH |s> = (1/2) UC UH (1 + (-1)^a (prod_j Z_j^{G_{pj} (1-v_j)} X_j^{G_{pj}^{v_j}}  )) |s>
        #then we apply the unitaries to |s> to get 
        # UC UH (|s> + (-1)^(k+a) |t>)
        k = self.a + (state.B[self.target] * np.uint8(1 + state.v) * state.s).sum(dtype=np.uint8) % np.uint8(2)
        t = np.uint8((state.B[self.target] * state.v) ^ state.s)

        if all(t == state.s):
            state.phase *= (1 + (-1)**k)/2
        else:
            phase, VCList, v, s = util.desuperpositionise(state.s, t, np.uint8(2*k % constants.UNSIGNED_4), state.v)
            for gate in VCList:
                gate.rightMultiplyC(state)
                
            state.phase *= phase / 2 # 2 since P = (I +- Z)/2
            state.v = v
            state.s = s
        return state
        
    def __str__(self):
        return "Pz({}, {})".format(self.target, self.a)

    def data(self):
        return "PZ", self.target, self.a
