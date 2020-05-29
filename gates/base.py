from abc import ABC, abstractmethod
class Gate(ABC):
    pass

class ComposableGate(Gate):
    """
    Composable gates are gates where it makes sense to apply other gates to a state after applying them
    I.e. they are gates which output a new state
    specifically every gate except MeasurementOutcomes gates are composable
    note that Clifford composible gates override the __or__ method so that they combine into CompositeCliffordGates to keep the applyCH and applyAG methods
    """
    def __or__(self, other:Gate) -> Gate:
        if isinstance(other, CompositeGate) and isinstance(self, CompositeGate):
            self.gates.extend(other.gates)
            return self
        elif isinstance(other, CompositeGate):
            other.gates.insert(0,self)
            return other
        elif isinstance(self, CompositeGate):
            self.gates.append(other)
            return self
        else:
            return Circuit([self,other])

class CompositeGate(ComposableGate):
    def __init__(self, gates=None):
        if gates == None:
            self.gates = []
        else:
            self.gates = gates

    def __str__(self):
        return "[" + ", ".join([gate.__str__() for gate in self.gates]) + "]"


class TGate(ComposableGate):
    def __init__(self, target):
        self.target = target
    def __str__(self):
        return "T({})".format(self.target)

