from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from stabstate import StabState
import constants
import measurement

class CliffordGate(ABC): #abstract base class
    """
    base class for both UnitaryCliffordGate and MeasurementOutcome
    """
    @abstractmethod
    def apply(self, state : StabState):
        pass

    
class UnitaryCliffordGate(CliffordGate):
    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # G (w UC UH) |s>
    @abstractmethod
    def apply(self, state : StabState) -> StabState:
        pass

    def __or__(self, other: CliffordGate) -> CliffordGate:
        if isinstance(other, measurement.MeasurementOutcome):
            other.gates.insert(0, self)
            return other
        elif isinstance(other, CompositeGate):
            other.gates.insert(0,self) # keep composite gates flat - we don't really want composite gates containing composite gates - note we also overide __or__ in CompositeGate
            return other
        else:
            return CompositeGate([self, other])
        
class CTypeCliffordGate(UnitaryCliffordGate): #abstract base class
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
        state.B[:,self.control] = state.B[:,self.control] ^ state.B[:,self.target]
        state.A[:,self.target] = state.A[:,self.target] ^ state.A[:,self.control]
        state.C[:,self.control] = state.C[:,self.control] ^ state.C[:,self.target]
        return state
    def leftMultiplyC(self, state: StabState) -> StabState:
        state.B[self.target] = state.B[self.target] ^ state.B[self.control] 
        state.A[self.control] = state.A[self.control] ^ state.A[self.target]
        state.C[self.control] = state.C[self.control] ^ state.C[self.target]
        state.g[self.control] = (state.g[self.control] + state.g[self.target] + 2* (state.C[self.control] @ state.A[self.target] % 2)) % constants.UNSIGNED_4
        return state

class CZGate(CTypeCliffordGate):
    """
    CZ gate with target and contol 
    """
    def __init__(self, target: int, control: int):
        self.target = target
        self.control = control
    def rightMultiplyC(self, state: StabState) -> StabState:
        state.C[:,self.control] = state.C[:,self.control] ^ state.A[:,self.target]
        state.C[:,self.target] = state.C[:,self.target] ^ state.A[:,self.control]
        state.g = (state.g + 2 * state.A[:,self.control] * state.A[:,self.target]) % constants.UNSIGNED_4
        return state
    def leftMultiplyC(self, state: StabState) -> StabState:
        state.C[self.control] = state.C[self.control] ^ state.B[self.target]
        state.C[self.target] = state.C[self.target] ^ state.B[self.control]
        return state
    
    
class HGate(UnitaryCliffordGate):
    """
    Hadamard gate with target
    """
    def __init__(self, target: int):
        self.target = target

    def apply(self, state: StabState) -> StabState:
        t = state.s + (state.B[self.target]* state.v)  %2
        u = (state.s + (state.A[self.target]*(1-state.v)) + (state.C[self.target]*state.v)) % 2
        alpha = (state.B[self.target]*np.uint8(1-state.v)*state.s).sum()
        beta = (state.C[self.target]*np.uint8(1-state.v)*state.s + state.A[self.target]*state.v*(state.C[self.target] + state.s)).sum()
        if all(t == u):
            state.phase = state.phase * ((-1)**alpha + (complex(0,1)**state.g[self.target])*(-1)**beta)/np.sqrt(2)
            return state
        else:
            tNeqUArray = t != u
            v0 = np.flatnonzero((state.v == 0) & tNeqUArray)
            v1 = np.flatnonzero((state.v == 1) & tNeqUArray)

            q = None
            VCList = []

            if len(v0) > 0:
                q = v0[0]
                VCList = [CXGate(control=q, target=i) for i in v0   if i != q]  + [CZGate(control=q,target=i) for i in v1] # this matches the first equation of page 20 - but I think it might be wrong
            else:
                q = v1.nonzero()[0]
                VCList = [CXGate(control=q, target=i) for i in v1 if i != q]

            y, z = None, None
            if t[q] == 1:
                y = np.copy(u)
                y[q] = ~y[q]
                z = np.copy(u)
            else: # t[q] == 0
                y = np.copy(t)
                z = np.copy(t)
                z[q] = (1-z[q]) %2
            #now we care about the state H_q^{v_q}  (|y_q> + i^delta |z_q>)
            #where y_q != z_q
            #lets put this in a standard form
            # i^w (|0> + i^(k) |1>)
            #by factorising out i^delta if necessary
            w = 0
            d = np.uint8(state.g[q] + 2*(alpha + beta) % constants.UNSIGNED_4)

            k = d
            if y[q] == 1: #so z[q] == 1
                w = d
                k = (4-d) % constants.UNSIGNED_4
            # now we write H^{v_q} (|0> + i^(k) |1>) = S^a H^b |c>
            a, b, c = None, None, None
            b = (state.v[q] + 1) %2

            if k == 0:
                a = 0
                c = 0
            elif k == 1:
                a = 1
                c = 0
            elif k == 2:
                a = 0
                c = 1
            elif k == 3:
                a = 1
                c = 1
                
            state.phase = state.phase * complex(0,1)**w
            state.s = y
            state.s[q] = c
            state.v[q] =  b % 2

            if a == 1:
                VCList.append(SGate(q))

            for gate in VCList:
                gate.rightMultiplyC(state)
            return state

        
class CompositeGate(CliffordGate):
    """
    just stores a list of gates and applies them one by one in its apply method
    """
    def __init__(self, gates=None):
        if gates == None:
            self.gates = []
        else:
            self.gates = gates

    def apply(self, state: StabState) -> StabState:
        for gate in self.gates:
            gate.apply(state)
        return state

    def __or__(self, other: CliffordGate) -> CliffordGate:
        if isinstance(other, measurement.MeasurementOutcome):
            other.gates = self.gates + other.gates
            return other
        elif isinstance(other, CompositeGate):
            # keep composite gates flat - we don't really want composite gates containing composite gates
            #note we also check for isinstance(other, CompositeGate) in CliffordGate.__or__
            self.gates.extend(other.gates)
        else:
            self.gates.append(other)
        return self
