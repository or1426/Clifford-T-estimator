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


def gaussian_eliminate(state):
    """
    We want to make the first column of F equal to 100...0 only using "allowed" operations
    all operations are inserted between Uc and Uh in the CH-form
    """

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
            
    #at this point add a pair of swaps so the first qubit has s = v = 0
    q = np.flatnonzero((state.v == 0) & (state.s == 0))[0]
    if q != 0:
        #left hand swap gets multiplied on to UC
        cliffords.CXGate(0, q).rightMultiplyC(state)
        cliffords.CXGate(q, 0).rightMultiplyC(state)
        cliffords.CXGate(0, q).rightMultiplyC(state)
        #right hand swap swaps the qubits
        state.v[[q,0]] = state.v[[0,q]]
        state.s[[q,0]] = state.s[[0,q]]
        
    #rref_state does a variant of the reduced column-echelon form calculation restricted to only doing cnots from qubits with s==v==0
    rcef_state(state)
    return state
