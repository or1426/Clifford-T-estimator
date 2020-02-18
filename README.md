# pscs
Phase sensitive Clifford simulator - implementation of https://arxiv.org/abs/1808.00128 section 4.1


## Construct computational basis states

Using the `StabState.basis` class method, pass either `N` an int, to construct the state |0...0> on n qubits, or `s`, a length N, one dimensional numpy array of `0`s and `1`s with `dtype=np.uint8` to construct the basis state given by that binary vector (e.g. `np.array([1,0,1], dtype=np,uint8)` becomes |101> = |1>|0>|1>). If both `N` and `s` are passed then `s` is truncated or extended (from the back) to length `N` as appropriate.

## Construct basic Clifford unitaries

The standard Clifford unitaries S, H, CX, and CZ are constructed by passing their target (and, if relevant control) qubit to the constructor. Examples
1. `SGate(0)` - constructs an S gate to be applied to the first qubit
1. `HGate(5)` - constructs an H gate to be applied to the fourth qubit
1. `CXGate(2,7)` - constructs a CX gate with target qubit 2 and control qubit 7
1. `CZGate(3,2)` - constructs a CX gate with target qubit 3 and control qubit 2

## Apply and compose Clifford unitaries

Of course we can apply unitaries to a state, the __or__ operator has been overloaded for this to provide syntax similar to unix pipes. `gate.apply(state)` is equivalent to `state | gate`, if state is a vector |v>, and the gate a unitary U then both result in U|v>. Note that the apply method (and pipe operator) both change the state in place, and return it so
```
s1 = StabState.basis(1)
s2 = s1 | HGate(0)
```
results in both `s1` and `s2` being equal to (|0> - |1>)/sqrt(2).

For convenience there is a composite gate class which simply stores a list of gates and applies them in order when it is applied to a state. For example
```
s1 | CompositeGate([HGate(0), CXGate(1,0)])
```
is equivalent to
```
s1 | HGate(0) | CXGate(1,0),
```
further, the pipe operation between gates creates a composite gate so `HGate(0) | CXGate(1,0)` is equivalent to  `CompositeGate([HGate(0), GXGate(1,0)])`. Possibly in the future the composite gate could do some optimisations (e.g. cancelling adjacent gates which are inverses of each other).

## Measurement outcomes

The `MeasurementOutcome` class exists to simulate a single outcome of an N-qubit observable which has 2^N 1-dimensional effects which are projectors onto the computational basis states. Applying it to a state gives the "overlap", examples
```
b00 = MeasurementOutcome(np.array([0,0], dtype=np.uint8)) # emulates the "bra" <00|
state = StabState.basis(2) # the "ket" |00>
overlap = state | H(0) | b00 # overlap equal to 1/sqrt(2)
```
or
```
b11 = MeasurementOutcome(np.array([1,1], dtype=np.uint8)) # emulates the "bra" <11|
state = StabState.basis(2) # the "ket" |00>
overlap = state | H(0) | b11 # overlap equal to 0 since this is <1|H|0> <1|0>.
```
We can also use the pipe notation to apply gates to a MeasurementOutcome, this means "when you have to compute an overlap, apply those gates, then compute the overlap". In particular
```
state | (gate | bra) == state | gate | bra == (state | gate) | bra,
```
where the first equality may be slightly wrong due to numerical precision and the second equality is because python evaluates expressions from the left to the right.
