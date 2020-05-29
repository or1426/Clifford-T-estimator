import cliffords
import numpy as np

def rcef_state(state):
    """
    Adapting the rref algorithm to work on CH-form and only apply to the columns we are allowed to touch
    also we're doing reduced column-echelon form not row
    """

    allowed_cols_bm = (state.s == 0) & (state.v == 0)
    allowed_cols = np.flatnonzero((state.s == 0) & (state.v == 0))
    
    h = 0 
    k = 0 

    m, n = state.N, len(allowed_cols)
    
    #part 1 - column echelon form
    while h < n and k < m:
        #Find the k-th pivot
        #look in row k for a pivot in a column we haven't been to yet
        fnz = np.flatnonzero( (state.F[k] == 1) & allowed_cols_bm )
        fnz = fnz[fnz >= allowed_cols[h]]
        if len(fnz) == 0:
            #No pivot in this row, pass to next row
            k = k+1
        else:
            pivot = fnz[0]
            if pivot != allowed_cols[h]:
                #we need to swap column h and pivot
                cliffords.CXGate(pivot, allowed_cols[h]).rightMultiplyC(state)
                cliffords.CXGate(allowed_cols[h], pivot).rightMultiplyC(state)
                cliffords.CXGate(pivot, allowed_cols[h]).rightMultiplyC(state)
            
            #for each column after the pivot
            #do a cnot from the pivot to that qubit to delete any 1s in the same row as the first 1 in the pivot column
            for i in range(allowed_cols[h] + 1, m):
                if state.F[k][i]:
                    cliffords.CXGate(target=i,control=allowed_cols[h]).rightMultiplyC(state)                    
            #Increase pivot row and column
            h = h + 1
            k = k + 1
            
    #part 2 - reduced column-echelon form
    for q in allowed_cols:
        #find leading 1
        fnz = np.flatnonzero(state.F[:,q]) # find non-zero elements in column q
        if len(fnz) > 0:
            k = fnz[0] #grab the first one, now all elements in this row, to the left of this should be made zero
            for j in range(q):
                if state.F[k][j]:
                    cliffords.CXGate(target=j, control = q).rightMultiplyC(state)
    
    return state


def fix(state, target):
    #we can't touch the qubits where v == 1

    #in general we don't want to do cnots from qubits where s == 1
    
    #but if we have two qubits where s == 1 we can do a cnot between them to make one of them have s == 0
    #in this way we can reduce the state to one where exactly one of the qubits has s == 1

    #Question - does it matter which qubit has s == 1??
    #lets assume it doesn't matter and see what happens

    s1Qubits = np.flatnonzero((state.v == 0) & (state.s == 1))
    if len(s1Qubits) > 1:
        for q in s1Qubits[1:]:
            #stick two cnots between s1Qubits[0] and s1Qubits[q]
            #cnot is its own inverse so this does not change the state
            #then the right cnot hits the s vector
            #the left cnot gets multiplied on to U_C

            cnot = cliffords.CXGate(control = s1Qubits[0], target = q)
            cnot.rightMultiplyC(state)
            state.s[q] = 0
            
    #at this point add a pair of swaps so the target qubit has s = v = 0
    q = np.flatnonzero((state.v == 0) & (state.s == 0))[0]
    if q != target:
        #left hand swap gets multiplied on to UC
        cliffords.CXGate(target, q).rightMultiplyC(state)
        cliffords.CXGate(q, target).rightMultiplyC(state)
        cliffords.CXGate(target, q).rightMultiplyC(state)
        #right hand swap swaps the qubits
        state.v[[q,target]] = state.v[[target,q]]
        state.s[[q,target]] = state.s[[target,q]]
    return state


def gaussian_eliminate(state,q=0):
    """
    We want to make the first column of F equal to 100...0 only using "allowed" operations
    all operations are inserted between Uc and Uh in the CH-form
    """
        
    #rref_state does a variant of the reduced column-echelon form calculation restricted to only doing cnots from qubits with s==v==0
    fix(state,q)
    rcef_state(state)
    
    #for r in range(state.N):
    #    er = np.zeros_like(state.v)
    #    er[r] = np.uint8(1)
    #    
    #    print(((er + h[r]) @ mat2)  % np.uint8(2))
    
    return state


def reduce_column(state,q):
    """
    Given a target column q of the state's F matrix, attempts to make it equal to e_q (the vector with a single 1 in the qth entry)
    only using allowed operations
    """

    if (state.G[q] * state.v != 0).any():
        return False
    
    fix(state,q)

    if (state.G[q]*state.s != 0).any():
        return False

    fnz = np.flatnonzero(state.G[q])
    if state.G[q][q] != 1:
        #swap qubit q and the first one from fnz
        cliffords.CXGate(fnz[0], q).rightMultiplyC(state)
        cliffords.CXGate(q, fnz[0]).rightMultiplyC(state)
        cliffords.CXGate(fnz[0], q).rightMultiplyC(state)
        #drop the first one from fnz
        fnz = fnz[1:]
        
    for j in fnz:
        if j != q:    
            cliffords.CXGate(control=j,target=q).rightMultiplyC(state)
    return state

def factor(state, q):
    if not reduce_column(state ,q):
        return False

    return state.delete_qubit(q)

def rcef(mat):
    m,n = mat.shape
    
    h = 0 #/* Initialization of the pivot row */
    k = 0 #/* Initialization of the pivot column */


    while h < n and k < m:
        #Find the k-th pivot
        #look in row k for a pivot in a column we haven't been to yet
        fnz = np.flatnonzero( (mat[k] == 1))
        fnz = fnz[fnz >= h]
        if len(fnz) == 0:
            #No pivot in this row, pass to next row
            k = k+1
        else:
            pivot = fnz[0]
            if pivot != h:
                mat[:, [pivot,h]] = mat[:,[h,pivot]]            
            #for each column after the pivot
            for i in range(h+ 1, n):
                if mat[k][i]:
                    mat[:,i] = (mat[:,i] + mat[:,h])%np.uint8(2)
            #Increase pivot row and column
            h = h + 1
            k = k + 1
            
    for q in range(n):
        #find leading 1
        fnz = np.flatnonzero(mat[:,q]) # find non-zero elements in column q
        if len(fnz) > 0:
            k = fnz[0] #grab the first one, now all elements in this row, to the left of this should be made zero
            for j in range(q):
                if mat[k][j]:
                    mat[:,j] = (mat[:,j] + mat[:,q])% np.uint8(2)
    return mat



def h_matrix(mat):
    #only really makes sense if mat is in reduced column echelon form

    n_rows, n_cols = mat.shape
    h = np.zeros((n_rows, n_rows), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            if mat[r][c]:
                fnz = np.flatnonzero(mat[:,c])
                #print(fnz)
                h[r, fnz[0]] +=  np.uint8(1)

    return h
