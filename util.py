import numpy as np
import constants
import cliffords

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
        VCList = [cliffords.CXGate(control=q, target=i) for i in v1 if i != q]
    
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
        k = d
        if y[q] == 1: #so z[q] == 1
            w = d
            k = (4-d) % constants.UNSIGNED_4
        # now we write H^{v_q} (|0> + i^(k) |1>) = sqrt(2) S^a H^b |c>
        a, b, c = None, None, None

        b = (v[q] + 1) %2

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

        phase = complex(0,1)**w *  np.sqrt(2) # fix normalisation factor 
        s = y
        s[q] = c
        v[q] =  b % 2 

        if a == 1:
            g = cliffords.SGate(q)
            VCList.append(g)

    return phase, VCList, v, s
