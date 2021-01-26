import numpy as np
import constants
import gates
import random
#import measurement
import itertools
#import myModule

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
        VCList = [gates.cliffords.CXGate(control=q, target=i) for i in v0   if i != q]  + [gates.cliffords.CZGate(control=q,target=i) for i in v1] 
    else:
        q = v1[0]
        VCList = [gates.cliffords.CXGate(control=i, target=q) for i in v1 if i != q]

    y, z = None, None
    if t[q] == 1:
        y = np.copy(u)
        y[q] = np.uint8((y[q] + 1) %2)
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
        g = gates.cliffords.SGate(q)
        VCList.append(g)
    return phase, VCList, v, s


def random_clifford_circuits(qubits, depth, N):
    #some Clifford gate constructors take two params and some take 1
    params_dict = {gates.cliffords.CXGate: 2,gates.cliffords.SGate: 1, gates.cliffords.CZGate: 2,gates.cliffords.HGate:1} #
    for _ in range(N):
        gs = random.choices(list(params_dict.keys()), k=depth)
        yield gates.cliffords.CompositeCliffordGate([g(*random.sample(range(qubits), k=params_dict[g])) for g in gs])
    
# def random_clifford_circuits_with_z_projectors(qubits, depth, N):
#     for target, a, circuit in zip(random.choices(range(qubits), k=N), random.choices(range(1), k=N), random_clifford_circuits(qubits, depth, N)):
#         yield circuit | measurement.PauliZProjector(target,a)

def random_clifford_circuits_with_bounded_T(qubits, depth, N, T,rng=None):
    #some Clifford gate constructors take two params and some take 1
    if rng==None:
        rng = random
    
    cliffords = [gates.cliffords.SGate, gates.cliffords.CZGate, gates.cliffords.CXGate, gates.cliffords.HGate]
    params_dict = {gates.cliffords.SGate: 1, gates.cliffords.CZGate: 2,gates.cliffords.CXGate: 2,gates.cliffords.HGate:1, gates.TGate:1} 
    count = 0
    for _ in range(N):
        gs = rng.choices(cliffords, k=depth)
        positions = rng.sample(range(len(gs)), T)
        for pos in positions:
            gs[pos] = gates.TGate

        yield gates.cliffords.CompositeCliffordGate([g(*rng.sample(range(qubits), k=params_dict[g])) for g in gs])

def random_clifford_circuits_with_fixed_T_positions(qubits, clifford_depth, N, T):
    for _ in range(N):
        circ = list(random_clifford_circuits(qubits, clifford_depth, 1))[0]

        for _ in range(T):
            circ | gates.TGate(random.randrange(qubits))
            circ | list(random_clifford_circuits(qubits, clifford_depth, 1))[0]
                    
        yield circ
        
def random_clifford_circuits_with_T(qubits, depth, N):
    #some Clifford gate constructors take two params and some take 1
    params_dict = {gates.cliffords.SGate: 1, gates.cliffords.CZGate: 2,gates.cliffords.CXGate: 2,gates.cliffords.HGate:1, gates.TGate:1} 
    count = 0
    while count < N:
        gs = random.choices(list(params_dict.keys()), k=depth)
        t = len([g for g in gs if g == gates.TGate])
        if t > 0:
            count += 1
            yield t, gates.cliffords.CompositeCliffordGate([g(*random.sample(range(qubits), k=params_dict[g])) for g in gs])

        
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


def sort_pauli_string(x,z):
    """
    Given nxn matrices x and z
    representing a Pauli string
    return the a (either 0 or 1) such that 
    prod_j prod_k X_k^{x_{jk}} Z_k^{z_{jk}} = (-1)^a prod_k prod_j Z_k^{z_{jk}} X_k^{x_{jk}} 
    note - order of product swapped and order of zs and xs flipped 
    """
    if len(z) == 0:
        return 0

    t = np.zeros_like(z[0])
    sign = 0
    for j in range(len(x)):
        t = t ^ z[j]        
        sign =  (sign + (t @ x[j])) % np.uint8(2)

    return sign

def find_asymetric_coords(M):
    """
    Given a square numpy array M return i,j such that M[i,j] != M[j,i]
    """
    for i in range(M.shape[0]):
        for j in range(i):
            if M[i,j] != M[j,i]:
                return i,j
    return None


def slowZ2ExponentialSum(M, L):
    """
    For testing purposes only
    This is exponentially slow
    """
    total = 0
    for tuple in itertools.product([0,1], repeat=len(L)):
        x = np.array(tuple)
        total += (-1)**( x @ (M @ x) + L@x)
    return total
        
def z2ExponentialSum(M, L):
    """
    Given a Z2 valued quadratic form
    q(x) = x. Mx + L.x mod 2
    compute sum_x (-1)^q(x) over all bitstrings x
    in cubic time
    """

    exponent_of_2 = 0
    exponent_of_minus_1 = 0
    while True:
        #we first seek indices i,j such that M_ij != M_ji
        coords = find_asymetric_coords(M)
        
        if coords == None:
            #M is symmetric
            #at this point the computation is trivial
            if (np.diag(M) == L).all():
                return ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L)))
            else:
                return 0
        else:
            i,j=coords
            mask = np.ones(len(L), dtype=bool)
            mask[[i,j]] = False
            
            m1 = (M[i] + M[:,i])[mask]
            m2 = (M[j] + M[:,j])[mask]

            mu1_consts = L[i] + M[i,i] % np.uint8(2)
            mu2_consts = L[j] + M[j,j] % np.uint8(2)

            M_else = M[mask][:,mask]
            L_else = L[mask]

            exponent_of_2 += 1
            exponent_of_minus_1 = (exponent_of_minus_1+mu1_consts*mu2_consts) % np.uint8(2)

            M = (M_else + np.outer(m1, m2)) % np.uint8(2)
            L = (L_else + mu1_consts*m2 + mu2_consts*m1) % np.uint8(2)
            
#@profile
def z2DoubleExponentialSum(M, L):
    """
    Given a Z2 valued quadratic form 
    q(x) = x. Mx + L.x mod 2
    where where L[-1] == 0
    compute sum_x (-1)^q(x) over all bitstrings x
    with both  L[-1] == 0 and L[-1] == 1
    in cubic time
    """
    exponent_of_2 = 0
    exponent_of_minus_1 = 0
    last_element_asymetric = False #if we hit an asymetric component of the matrix with i = len(L) - 1 we make this true
    mu1_consts = None
    mu2_consts = None
    mask = np.ones(len(L), dtype=bool)
    u2 = np.uint8(2)
    while True:
        #we first seek indices i,j such that M_ij != M_ji
        #coords = find_asymetric_coords(M)
        coords = find_asymetric_coords(M)
        
        #print(coords)
        if coords == None:
            #M is symmetric
            #at this point the computation is trivial
            if last_element_asymetric:
                if (np.diag(M) == L).all():
                    new_exponent_of_minus_1 = (exponent_of_minus_1 + mu1_consts*mu2_consts + (mu1_consts+1)*mu2_consts) %np.uint8(2)
                    return ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L))), ((-1)**new_exponent_of_minus_1) * (2**(exponent_of_2+len(L)))
                else:
                    return 0, 0
                
            else:
                if (np.diag(M)[:-1] == L[:-1]).all():
                    if (np.diag(M) == L).all():
                        #print(np.diag(M))
                        return ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L))), 0
                    else:
                        #print(np.diag(M))
                        return 0, ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L)))
                        
                else:
                    return 0, 0
        else:
            i,j=coords
            if i == len(L) -1: # note that j < i so only i can be the last index
                #we're about to delete the last remaining row and column in M
                #since the coordinates returned by find_asymetric_coords are ordered
                last_element_asymetric = True

            mask[i] = False
            mask[j] = False
            
            m1 = (M[i] + M[:,i])[mask] % np.uint8(2)
            m2 = (M[j] + M[:,j])[mask] % np.uint8(2)
            
            mu1_consts = L[i] + M[i,i] % np.uint8(2)
            mu2_consts = L[j] + M[j,j] % np.uint8(2)

            M = M[mask][:,mask]
            L = L[mask]

            exponent_of_2 += 1
            exponent_of_minus_1 = (exponent_of_minus_1+mu1_consts*mu2_consts) % np.uint8(2)

                         
            #M[np.bool_(m1)] += m2
            #print(m1)
            #print(m2)
            #M = (M + np.outer(m1, m2)) % np.uint8(2)
            myModule.add_outer_product(M, m1, m2);
            #M %= 2
            
            #L = (L  + mu1_consts*m2 + mu2_consts*m1) % 2
            if mu1_consts == 1:
                L += m2
            if mu2_consts == 1:
                L += m1
            L %= 2
            
            mask[i] = True
            mask[j] = True
            mask = mask[:-2]
        
#@profile            
def z2DoubleExponentialSum2(M, L):
    """
    Given a Z2 valued quadratic form 
    q(x) = x. Mx + L.x mod 2
    where where L[-1] == 0
    compute sum_x (-1)^q(x) over all bitstrings x
    with both  L[-1] == 0 and L[-1] == 1
    in cubic time
    """
    exponent_of_2 = 0
    exponent_of_minus_1 = 0
    last_element_asymetric = False #if we hit an asymetric component of the matrix with i = len(L) - 1 we make this true
    mu1_consts = None
    mu2_consts = None
    mask = np.zeros(len(L), dtype=bool)
    u2 = np.uint8(2)
    killed = 0
    while True:
        #we first seek indices i,j such that M_ij != M_ji
        #coords = find_asymetric_coords(M)
        coords = find_asymetric_coords(M)
        
        ##print(coords)
        if coords == None:
            #M is symmetric
            #at this point the computation is trivial
            #print("py last_element_asymetric", last_element_asymetric)
            #print("py", exponent_of_minus_1, exponent_of_2, killed, mu2_consts, last_element_asymetric)
            #print(M)
            #print(L)
            
            if last_element_asymetric:
                if (np.diag(M) == L).all():
                    #print("p1")
                    new_exponent_of_minus_1 = (exponent_of_minus_1 + mu1_consts*mu2_consts + (mu1_consts+1)*mu2_consts) %np.uint8(2)
                    return ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L)-killed)), ((-1)**new_exponent_of_minus_1) * (2**(exponent_of_2+len(L)-killed))
                else:
                    #print("p2")
                    return 0, 0
                
            else:
                if (np.diag(M)[:-1] == L[:-1]).all():
                    if (np.diag(M) == L).all():
                        ##print(np.diag(M))
                        #print("p3")
                        return ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L)-killed)), 0
                    else:
                        ##print(np.diag(M))
                        #print("p4")
                        return 0, ((-1)**exponent_of_minus_1) * (2**(exponent_of_2+len(L)-killed))
                        
                else:
                    #print("p5")
                    return 0, 0
        else:
            i,j=coords
            ##print("[{},{}]".format(i,j))
            ##print(M)
            if i == len(L) - 1: # note that j < i so only i can be the last index
                #we're about to delete the last remaining row and column in M
                #since the coordinates returned by find_asymetric_coords are ordered
                last_element_asymetric = True

            killed += 2
            ##print(M)
            m1 = (M[i] + M[:,i]) % np.uint8(2)
            m1[i] = 0
            m1[j] = 0
            m2 = (M[j] + M[:,j]) % np.uint8(2)
            m2[i] = 0
            m2[j] = 0
            #print("py m1", m1)
            #print("py m2", m2)
            mu1_consts = (L[i] + M[i,i]) % np.uint8(2)
            mu2_consts = (L[j] + M[j,j]) % np.uint8(2)
            
            #print("pyconsts:", mu1_consts, mu2_consts)
            #print(M)
            #print("pyL", L)
            M[i] = 0
            M[j] = 0
            M[:,i] = 0
            M[:,j] = 0

            L[i] = 0
            L[j] = 0

            exponent_of_2 += 1
            exponent_of_minus_1 = (exponent_of_minus_1+ mu1_consts*mu2_consts) % np.uint8(2)

                         
            #M[np.bool_(m1)] += m2
            ##print(m1)
            ##print(m2)
            #M = (M + np.outer(m1, m2)) % np.uint8(2)
            myModule.add_outer_product(M, m1, m2);
            #M %= 2
            
            #L = (L  + mu1_consts*m2 + mu2_consts*m1) % 2
            if mu1_consts == 1:
                L += m2
            if mu2_consts == 1:
                L += m1
            L %= 2
            
            #mask[i] = False
            #mask[j] = False

        
            
