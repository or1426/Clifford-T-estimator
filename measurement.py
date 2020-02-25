import numpy as np
from stabstate import StabState
import constants
import cliffords


class MeasurementOutcome(cliffords.CliffordGate):
    def __init__(self, x: np.ndarray, gates=None): # if gates is not None its a list of Clifford unitaries we apply to the state before computing the overlap 
        self.x = np.uint8(x)
        if gates == None:
            self.gates = []

    #given out outcome <v|, where v is a binary vector of length 2^n
    #and an n-qubit stabiliser state w UC UH |s> = 
    #calculate <v|s>
    def apply(self, state: StabState) -> complex:
        for gate in self.gates:
            gate.apply(state)
        
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

# class PauliZProjector(cliffords.CliffordGate):
#     """
#     Class to model a projector of the form 
#     (I + (-1)^a_{0} z_0^{b_0})/2 * (I + (-1)^a_{1} z_1^{b_1})/2 * ... (I + (-1)^a_{n-1} z_{n-1}^{b_n-1})/2 
#     where a and b are length n vectors of bits and z_i is the unitary acting as Pauli-z on qubit i, and the identity on the others
#     """
#     def __init__(self, a:np.ndarray, b:np.ndarray):
#         self.a = a # p is a vector of bits, and P[j] is the power to which z_j is to be raised
#         self.b = b

#     """
#     Applying a projector to a state results in a "state" which is not normalised
#     the normalisation factor will be kept in the phase of the stabaliser state
#     """
#     def apply(self, state: StabState) -> StabState:
#         #commuting a Pauli z through UC results in a tensor product of a bunch of Pauli zs based on equation 43
#         pass
        
