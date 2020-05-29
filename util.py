import numpy as np
import constants
from gates import cliffords
import random
import measurement

def a2str(a):
    """
    formatting for 1 or 2 dimensional numpy arrays of booleans
    """
    if len(a.shape) == 1:
        return "".join(map(str, a))
    elif len(a.shape) == 2:
        return "\n".join(map(lambda row: "".join(map(str, row)), a))

def pprint(a):
    """
    formatting for 1 or 2 dimensional numpy arrays of booleans
    """
    print(a2str(a))


    
def desuperpositionise(t, u, d, v):
    """
    given two bit-vectors t and u, which are not equal and a state we know is of the form 
    UH (|t> + i^d |u>)
    where UH is a tensor product of Hadamard gates, H_0^v[0] H_1^v[1] .. H_{n-1}^v[n-1] 
    choose a q and compute
    phase UC VC UH (|x> + i^d |y>)
    such that VC is a C-type Clifford gate, and x[q] != y[q], but x[i] = y[i] for i != q
    then return 
    phase, VC, v', |s> 
    where phase is a complex phase, VC is a list of C-type gates and v and s are bit-vectors such that
    phase (product of VC) (product of H_i^v'[i]) |s> = UH (|t> + i^d |u>)
    See proposition 4 of arXiv:1808.00128
    """
    tNeqUArray = t != u

    if all(tNeqUArray == 0):
        raise ValueError("t and u should differ: {}, {}".format(t,u))
    v0 = np.flatnonzero((v == 0) & tNeqUArray)
    v1 = np.flatnonzero((v == 1) & tNeqUArray)

    q = None
    VCList = []

    if len(v0) > 0:
        q = v0[0]
        VCList = [cliffords.CXGate(control=q, target=i) for i in v0   if i != q]  + [cliffords.CZGate(control=q,target=i) for i in v1] 
    else:
        q = v1[0]
        VCList = [cliffords.CXGate(control=i, target=q) for i in v1 if i != q]

    y, z = None, None
    if t[q] == 1:
        y = np.copy(u)
        y[q] = np.uint((y[q] + 1) %2)
        z = np.copy(u)
    else: # t[q] == 0
        y = np.copy(t)
        z = np.copy(t)
        z[q] = np.uint8((1+z[q]) %2)

    #now we care about the state H_q^{v_q}  (|y_q> + i^delta |z_q>)
    #where y_q != z_q
    #lets put this in a standard form
    # i^w (|0> + i^(k) |1>)
    #by factorising out i^delta if necessary
    w = np.uint8(0)
    k = np.uint8(d)
    
    if y[q] == 1: #so z[q] == 1
        w = np.uint8(d)
        k = np.uint8((4-d) % constants.UNSIGNED_4)
    # now we write H^{v_q} (|0> + i^(k) |1>) = sqrt(2) S^a H^b |c>

    a, b, c = None, None, None
    phase = complex(0,1)**w * np.sqrt(2)

    #is there a better way to write this?
    if v[q] == 0:
        b = 1
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
    else: # v[1] == 1
        if k == 0:
            a = 0
            b = 0
            c = 0
        elif k == 1:
            a = 1
            b = 1
            c = 1
            phase *= complex(1,1)/np.sqrt(2)
        elif k == 2:
            a = 0
            b = 0
            c = 1
        elif k == 3:
            a = 1
            b = 1
            c = 0
            phase *= complex(1,-1)/np.sqrt(2)
    if k != 0 and k != 1 and k != 2 and k != 3:
        print(k)
        print(k.dtype)
        print(d)
        
    s = y
    s[q] = c
    v[q] =  b % 2 
    
    if a == 1:
        g = cliffords.SGate(q)
        VCList.append(g)

    return phase, VCList, v, s


def random_clifford_circuits(qubits, depth, N):
    #some Clifford gate constructors take two params and some take 1
    params_dict = {cliffords.SGate: 1, cliffords.CZGate: 2,cliffords.CXGate: 2,cliffords.HGate:1} 
    for _ in range(N):
        gates = random.choices(list(params_dict.keys()), k=depth)
        yield cliffords.CompositeGate([gate(*random.sample(range(qubits), k=params_dict[gate])) for gate in gates])
    
def random_clifford_circuits_with_z_projectors(qubits, depth, N):
    for target, a, circuit in zip(random.choices(range(qubits), k=N), random.choices(range(1), k=N), random_clifford_circuits(qubits, depth, N)):
        yield circuit | measurement.PauliZProjector(target,a)

        
def rref(mat):
    m,n = mat.shape
    
    h = 0 #/* Initialization of the pivot row */
    k = 0 #/* Initialization of the pivot column */

    while h < n and k < m:
        #Find the k-th pivot
        #look in column h for a pivot in a row we haven't been to yet
        fnz = np.flatnonzero( (mat[:,h] == 1))
        fnz = fnz[fnz >= k]
        if len(fnz) == 0:
            #No pivot in this column, pass to next column
            h = h+1
        else:
            pivot = fnz[0]
            if pivot != k:
                mat[[pivot,k]] = mat[[k,pivot]]            
            #for each row after the pivot
            for i in range(k + 1, n):
                if mat[i][h]:
                    mat[i] = (mat[i] + mat[k])%np.uint8(2)
            #Increase pivot row and column
            h = h + 1
            k = k + 1
            
    for q in range(n):
        #find leading 1
        fnz = np.flatnonzero(mat[q]) # find non-zero elements in row q
        if len(fnz) > 0:
            h = fnz[0] #grab the first one, now all elements in this column, below this should be made zero
            for j in range(q):
                if mat[j][h]:
                    mat[j] = (mat[j] + mat[q])% np.uint8(2)
    return mat
