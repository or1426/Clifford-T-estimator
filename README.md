# Clifford+T estimator
Clifford+T estimator - estimate a single probability of an n qubit quantum circuit consisting of Clifford gates, arbitrary single-qubit diagonal gates gates and a w qubit computational basis measurement .

This is an implementation of the algorithms reported in https://arxiv.org/abs/2101.12223

# Installation

You will require
* python3 
  * numpy
  * (optional) qiskit
* GCC (or equivalent C compiler - see notes below)

The setup.py file will compule and link the C code for you. A sufficient setup is (when executed from the directory containing the setup.py file):

`python3 ./setup.py build_ext --inplace`

This setup will compile the code and build the extension module in the current directory.

## Installation notes

The code has only been tested with the following versions
  * Linux 4.19.0-13-amd64 SMP Debian 4.19.160-2 (2020-11-28) x86_64
  * gcc (Debian 8.3.0-6) 8.3.0
  * Python 3.7.3
  * numpy version 1.18.5

We make use of some gcc extensions, so other C compilers are not recommended.

In the estimate algorithm code we store [CH-forms](https://quantum-journal.org/papers/q-2019-09-02-181/) as binary matrices where a row of a matrix is a single C object of type `uint_bitarray_t`. If you require fewer than 64 qubits in the estimate code then you should leave the typedef

`typedef uint_fast64_t uint_bitarray_t;`

if you require between 64 and 123 qubits in the estimate code then  you should replace this with the typedef 

`typedef unsigned __int128 uint_bitarray_t;`

Doing so will make the code substantially slower. Note that the number of qubits used in the estimate code is upper bounded by the number of non-Clifford gates in your circuit (and may be substantially less in some cases).


# Basic usage

We provide implementations of the following algorithms, as defined in https://arxiv.org/abs/2101.12223

* Compress - which is insensitive to the difference between different single-qubit phase gates
* Compute - an implementation suitable only for pi/4 T gates
* Estimate - an implementation suitable only for pi/4 T gates
* Estimate - an implementation suitable for arbitrary phase gates

We also provide some routines to make using these more convenient. 
