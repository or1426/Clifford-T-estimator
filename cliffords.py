import numpy as np
from abc import ABC, abstractmethod
from stabstate import StabState
import constants

class CliffordGate(ABC): #abstract base class
    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # G (w UC UH) |s>
    @abstractmethod
    def apply(self, state : StabState) -> StabState:
        pass

class CTypeCliffordGate(CliffordGate): #abstract base class
    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # w (UC G) UH |s>
    @abstractmethod
    def rightMultiplyC(self, state : StabState) -> StabState:
        pass

    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # w (G UC) UH |s>
    @abstractmethod
    def leftMultiplyC(self, state : StabState) -> StabState:
        pass

    #applying C type gates is easy
    #just left-multiply it on to UC
    def apply(self, state : StabState) -> StabState:
        self.leftMultiplyC(state)
        return state
    
class SGate(CTypeCliffordGate):
    """
    S gate applied to qubit target
    """
    def __init__(self, target: int):
        self.target = target
        
    def rightMultiplyC(self, state : StabState) -> StabState:
        state.C[:,self.target] = state.C[:, self.target] ^ state.A[:, self.target]
        state.g = (state.g - state.A[:, self.target]) % constants.UNSIGNED_4
        return state

    def leftMultiplyC(self, state : StabState) -> StabState:
        state.C[self.target] = state.C[self.target] ^ state.B[self.target]
        state.g[self.target] = (state.g[self.target] - 1) % constants.UNSIGNED_4
        return state

class CXGate(CTypeCliffordGate):
    """
    CX gate with target and control 
    """
    def __init__(self, target: int, control: int):
        self.target = target
        self.control = control
    def rightMultiplyC(self, state: StabState) -> StabState:
        state.B[:,self.target] = state.B[:,self.target] ^ state.B[:,self.control]
        state.A[:,self.control] = state.A[:,self.control] ^ state.A[:,self.target]
        state.C[:,self.target] = state.C[:,self.target] ^ state.C[:,self.control]
        return state
    def leftMultiplyC(self, state: StabState) -> StabState:
        state.B[self.control] = state.B[self.control] ^ state.B[self.target] 
        state.A[self.target] = state.A[self.target] ^ state.A[self.control]
        state.C[self.target] = state.C[self.target] ^ state.C[self.control]
        state.g[self.target] = (state.g[self.target] + state.g[self.control] + 2* (state.C[self.target] @ state.A[self.control] % 2)) % constants.UNSIGNED_4
        return state

class CZGate(CTypeCliffordGate):
    """
    CZ gate with target and contol 
    """
    def __init__(self, target: int, control: int):
        self.target = target
        self.control = self.control
    def rightMultiplyC(self, state: StabState) -> StabState:
        state.C[:,self.target] = state.C[:,self.target] ^ state.A[:,self.control]
        state.C[:,self.control] = state.C[:,self.control] ^ state.A[:,self.target]
        state.g = (state.g + 2 * state.A[:,self.target] * state.A[:,self.control]) % constants.UNSIGNED_4
        return state
    def leftMultiplyC(self, state: StabState) -> StabState:
        state.C[self.target] = state.C[self.target] ^ state.B[self.control]
        state.C[self.control] = state.C[self.control] ^ state.B[self.target]
        return state
    
    
class HGate(CliffordGate):
    """
    Hadamard gate with target
    """
    def __init__(self, target: int):
        self.target = target

    def apply(self, state: StabState) -> StabState:
        t = (state.s ^ state.B[target]) * state.v
        u = state.s ^ (state.A[target]*(~state.v)) ^ (state.C[target]*state.v)

        alpha = (state.B[target]*(~v)*state.s).sum()
        beta = (state.C[target]*(~v)*state.s + state.A[target]*state.v*(state.C[target] + state.s)).sum()
        
        if all(t == u):
            state.phase = state.phase * ((-1)**alpha + (compex(0,1)**state.g[target])*(-1)**beta)/np.sqrt(2)
            return state
        else:
            tNeqUArray = np.argwhere(t != u)            
            v0 = np.argwhere((state.v == 0) & tNeqUArray)
            v1 = np.argwhere((state.v == 1) & tNeqUArray)
            q = None
            VCList = []
            
            if v0.any():
                q = v0.nonzero()[0]
                VCList = [CXGate(q, i) for i in v0   if i != q]  + [CZGate(q,i) for i in v1] # this matches the first equation of page 20 - but I think it might be wrong
            else:
                q = v1.nonzero()[0]
                VCList = [CXGate(q, i) for i in v1 if i != q]

            y, z = None, None
            if t[q] == 1:
                y = np.copy(u)
                y[q] = ~y[q]
                z = np.copy(u)
            else: # t[q] == 0
                y = np.copy(t)
                z = np.copy(t)
                z[q] = ~z[q]
                
            #now we care about the state H_q^{v_q}  (|y_q> + i^delta |z_q>)
            #where y_q != z_q
            #lets put this in a standard form
            # i^w (|0> + i^(k) |1>)
            #by factorising out i^delta if necessary
            
            w = 0
            d = state.g[q] + 2*(alpha + beta) % constants.UNSIGNED_4
            k = d
            if y[q] == 1: #so z[q] == 1
                w = d
                k = (4-d) % constants.UNSIGNED_4 

            # now we write H^{v_q} (|0> + i^(k) |1>) = S^a H^b |c>
            a, b, c = None, None, None
            if v[q] == 0: 
                if k == 0:
                    a = 0
                    b = 1
                    c = 0
                elif k == 1:
                    a = 1
                    b = 1
                    c = 0
                elif k == 2:
                    a = 1
                    b = 0
                    c = 1
                elif k == 3:
                    a = 1
                    b = 1
                    c = 1
            else: # v[q] == 1
                if k == 0:
                    a = 0
                    b = 0
                    c = 0
                elif k == 1:
                    a = 1
                    b = 1
                    c = 1
                elif k == 2:
                    a = 0
                    b = 0
                    c = 1
                elif k == 3:
                    a = 1
                    b = 1
                    c = 0
            state.phase = state.phase * complex(0,1)**w
            state.v = y
            state.s[q] = c
            state.v[q] = b
            if a == 1:
                VCList.append(SGate(q))

            for gate in VCList:
                gate.rightMultiplyC(state)
            return state
