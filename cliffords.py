from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from chstate import CHState
from agstate import AGState
import constants
import measurement



import util

class CliffordGate(ABC): #abstract base class
    """
    base class for both UnitaryCliffordGate and MeasurementOutcome
    """
    @abstractmethod
    def apply(self, state : CHState):
        pass

    @abstractmethod
    def applyAG(self, state: AGState):
        pass

    
class UnitaryCliffordGate(CliffordGate):
    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # G (w UC UH) |s>
    @abstractmethod
    def apply(self, state : CHState) -> CHState:
        pass
    @abstractmethod
    def applyAG(self, state: AGState) ->AGState:
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
    @abstractmethod
    def data(self):
        pass
        
class CTypeCliffordGate(UnitaryCliffordGate): #abstract base class
    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # w (UC G) UH |s>
    @abstractmethod
    def rightMultiplyC(self, state : CHState) -> CHState:
        pass

    # given a state that looks like
    # w UC UH |s>
    # compute the CH form of the state state
    # w (G UC) UH |s>
    @abstractmethod
    def leftMultiplyC(self, state : CHState) -> CHState:
        pass

    #applying C type gates is easy
    #just left-multiply it on to UC
    def apply(self, state : CHState) -> CHState:
        self.leftMultiplyC(state)
        return state
    
class SGate(CTypeCliffordGate):
    """
    S gate applied to qubit target
    """
    def __init__(self, target: int):
        self.target = target

    def rightMultiplyC(self, state : CHState) -> CHState:
        state.C[:,self.target] = state.C[:, self.target] ^ state.A[:, self.target]
        state.g = np.uint8(state.g - state.A[:, self.target]) % constants.UNSIGNED_4
        return state

    def leftMultiplyC(self, state : CHState) -> CHState:
        state.C[self.target] = state.C[self.target] ^ state.B[self.target]
        state.g[self.target] = (state.g[self.target] + np.uint8(3)) % constants.UNSIGNED_4
        return state

    def __str__(self):
        return "S({})".format(self.target)
    def data(self):
        return "S", self.target

    def applyAG(self, state : AGState) -> AGState:
        state.r = state.r ^ (state.x[:,self.target]*state.z[:,self.target])
        state.z[:,self.target] = state.z[:,self.target] ^ state.x[:,self.target]
        return state

class XGate(UnitaryCliffordGate):
    """
    X gate applied to qubit target
    """
    def __init__(self, target: int):
        self.target = target


    def apply(self, state : CHState) -> CHState:
        alpha = np.uint8((state.G[self.target]*(1-state.v)*state.s).sum() % np.uint8(2))
        state.s = np.uint8(state.s + state.G[self.target]*state.v % np.uint8(2))
        state.phase *= (-1)**alpha
        
        return state
    def applyAG(self, state : AGState) -> AGState:
        state.r = state.r ^ state.z[:,self.target]
        return state

    def __str__(self):
        return "X({})".format(self.target)
    def data(self):
        return "X", self.target

        
class CXGate(CTypeCliffordGate):
    """
    CX gate with target and control 
    """
    def __init__(self, target: int, control: int):
        self.target = target
        self.control = control
        
    def rightMultiplyC(self, state: CHState) -> CHState:
        state.B[:,self.control] = state.B[:,self.control] ^ state.B[:,self.target]
        state.A[:,self.target] = state.A[:,self.target] ^ state.A[:,self.control]
        state.C[:,self.control] = state.C[:,self.control] ^ state.C[:,self.target]
        return state
    
    def leftMultiplyC(self, state: CHState) -> CHState:
        state.g[self.control] = (state.g[self.control] + state.g[self.target] + np.uint8(2) * (state.C[self.control] @ state.A[self.target] )) % constants.UNSIGNED_4
        state.B[self.target] = state.B[self.target] ^ state.B[self.control] 
        state.A[self.control] = state.A[self.control] ^ state.A[self.target]
        state.C[self.control] = state.C[self.control] ^ state.C[self.target]
        return state

    def applyAG(self, state : AGState) -> AGState:
        state.r = (state.r + (state.x[:,self.control]*state.z[:,self.target]) * ((state.x[:,self.target]+state.z[:,self.control]+np.uint8(1)) %np.uint8(2)))%np.uint8(2)
        state.x[:,self.target] = state.x[:,self.target] ^ state.x[:,self.control]
        state.z[:,self.control] = state.z[:,self.control] ^ state.z[:,self.target]
        
        return state

    def __str__(self):
        return "CX({}, {})".format(self.target, self.control)

    def data(self):
        return "CX", self.target, self.control

class CZGate(CTypeCliffordGate):
    """
    CZ gate with target and contol 
    """
    def __init__(self, target: int, control: int):
        self.target = target
        self.control = control
    def rightMultiplyC(self, state: CHState) -> CHState:
        state.C[:,self.control] = state.C[:,self.control] ^ state.A[:,self.target]
        state.C[:,self.target] = state.C[:,self.target] ^ state.A[:,self.control]
        state.g = (state.g + 2 * state.A[:,self.control] * state.A[:,self.target]) % constants.UNSIGNED_4
        return state
    def leftMultiplyC(self, state: CHState) -> CHState:
        state.C[self.control] = state.C[self.control] ^ state.B[self.target]
        state.C[self.target] = state.C[self.target] ^ state.B[self.control]
        return state

    def applyAG(self, state: AGState) -> AGState:
        return HGate(self.target).applyAG(CXGate(self.target, self.control).applyAG(HGate(self.target).applyAG(state)))

    
    def __str__(self):
        return "CZ({}, {})".format(self.target, self.control)
    
    def data(self):
        return "CZ", self.target, self.control

    
class HGate(UnitaryCliffordGate):
    """
    Hadamard gate with target
    """
    def __init__(self, target: int):
        self.target = target

    def apply(self, state: CHState) -> CHState:
        t = state.s ^ (state.B[self.target]* state.v) 
        u = (state.s ^ (state.A[self.target]*np.uint8(1-state.v)) ^ (state.C[self.target]*state.v)) 
        alpha = (state.B[self.target]*np.uint8(1-state.v)*state.s).sum()
        beta = (state.C[self.target]*np.uint8(1-state.v)*state.s + state.A[self.target]*state.v*(state.C[self.target] + state.s)).sum()
        
        if all(t == u):
            state.s = t
            state.phase = state.phase * ((-1)**alpha + (complex(0,1)**state.g[self.target])*(-1)**beta)/np.sqrt(2)
            return state
        else:
            phase, VCList, v, s = util.desuperpositionise(t, u, (state.g[self.target] + 2 * (alpha+beta)) % constants.UNSIGNED_4 , state.v)
            state.phase *= phase
            state.phase *= (-1)**alpha / np.sqrt(2) # sqrt(2) since H = (X + Z)/sqrt(2)
            state.v = v
            state.s = s

            for gate in VCList:
                gate.rightMultiplyC(state)
            
            return state

    def applyAG(self, state : AGState) -> AGState:
        state.r = (state.r + state.x[:,self.target]*state.z[:,self.target]) % np.uint8(2)
        state.x[:,self.target], state.z[:,self.target] = state.z[:,self.target].copy(), state.x[:,self.target].copy()
        
        return state
    
    def __str__(self):
        return "H({})".format(self.target)

    def data(self):
        return "H", self.target

        
class CompositeGate(CliffordGate):
    """
    just stores a list of gates and applies them one by one in its apply method
    """
    def __init__(self, gates=None):
        if gates == None:
            self.gates = []
        else:
            self.gates = gates

    def apply(self, state: CHState) -> CHState:
        for gate in self.gates:
            gate.apply(state)
        return state
    
    def applyAG(self, state: AGState) -> AGState:
        for gate in self.gates:
            gate.applyAG(state)
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

    def __str__(self):
        return "[" + ", ".join([gate.__str__() for gate in self.gates]) + "]"


class SwapGate(CTypeCliffordGate):
    """
    Swaps qubits a and b
    Can be implemented as three CNOTS but just swapping the elements of our matrices and vectors is more efficient
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def leftMultiplyC(self, state: CHState) -> CHState:
        permutation = list(range(state.N))
        permutation[self.a] = self.b
        permutation[self.b] = self.a

        state.A = state.A[permutation][:,permutation]
        state.B = state.B[permutation][:,permutation]
        state.C = state.C[permutation][:,permutation]
        state.g = state.g[permutation]
        state.v = state.v[permutation]
        state.s = state.s[permutation]
        
        return state
        
    def rightMultiplyC(self, state: CHState) -> CHState:
        state.v = state.v[permutation]
        state.s = state.s[permutation]
        return state

    def applyAG(self, state: AGState) -> AGState:
        return CXGate(self.a, self.b).applyAG(CXGate(self.b, self.a).applyAG(CXGate(self.a, self.b).applyAG(state)))
    
    def data(self):
        return "SWAP", self.a, self.b 
