from dataclasses import dataclass
@dataclass
class AGState:
    n : int # number of qubits
    k : int # number of stabilizers 
    x : np.ndarray
    z : np.ndarray
    r : np.ndarray
    
    @classmethod
    def __init__(self, n, k=None):
        if k == None:
            k = n
        self.n = n
        self.k = k
        self.x = np.zeros((self.k,self.n), dtype=np.uint8)
        self.z = np.eye(self.k,self.n, dtype=np.uint8)
        self.r = np.zeros(self.k, dtype=np.uint8)
    
    def _rowToStr(row):
        return "".join(map(str,row))

    def row2Str(self, i):
        #each row is a single stabiliser
        s = ""
        if self.r[i]:
            s += "-"
        else:
            s += "+"
        for x, z in zip(self.x[i], self.z[i]):
            if x==0 and z==0:
                s += "I"
            if x==1 and z==0:
                s += "X"
            if x==0 and z==1:
                s += "Z"
            if x==1 and z==1:
                s += "Y"
        return s
                
    def _g(x1,z1,x2,z2):
        return x1*z1*(z2 - np.int8(x2)) + x1*((z1 + 1) % np.uint8(2))*z2*(2*np.int8(x2) -1) + ((x1 + 1) % np.uint8(2)) *z1*x2*(1-2*np.int8(z2))

    def _g2(x1,z1,x2,z2):
        if x1 == 1 and z1 == 1:
            return z2 - np.int8(x2)
        elif x1 == 1 and z1 == 0:
            return z2*(2*x2-1)
        elif x1 == 0 and z1 == 1:
            return x2*(1-2*z2)
        else:
            return 0
        
    def rowsum(self, h, i):
        s = 2*self.r[h] + 2*self.r[i] + AGState._g(self.x[i], self.z[i], self.x[h], self.z[h]).sum()
        s = s % np.uint8(4)        
        
        self.r[h] = s//np.uint8(2)
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]
            
    def rowswap(self, h, i):
        self.x[[h,i]] = self.x[[i,h]]
        self.z[[h,i]] = self.z[[i,h]]
        self.r[[h,i]] = self.r[[i,h]]


    def H(self, q):
        for i in range(self.k):
            self.r[i] ^= self.x[i][q] * self.z[i][q] 
            self.x[i][q], self.z[i][q] = self.z[i][q], self.x[i][q]
    def S(self, q):
        for i in range(self.k):
            self.r[i] ^= self.x[i][q] * self.z[i][q]
            self.z[i][q] ^= self.x[i][q]

    def CX(self, a, b):
        for i in range(self.k):
            self.r[i] ^= self.x[i][a] * self.z[i][b] * (self.x[i][b] ^ self.z[i][a] ^ 1)
            self.x[i][b] ^= self.x[i][a]
            self.z[i][a] ^= self.z[i][b]
            
    def CZ(self, a,b):
        self.H(b)
        self.CX(a,b)
        self.H(b)

    def phaseless_cb_inner_product(self):
        """
        here we do something a bit strange
        we compute the biggest inner product between our state an a computational basis state
        """
        rank = 0
        for col in range(self.n):
            pivot = -1
            for row in range(rank, self.k):
                if self.x[row][col] == 1:
                    pivot = row
                    break
            if pivot >= 0:
                self.rowswap(rank, pivot)
                for i in range(0, self.k):
                    if i != rank and self.x[i][col] == 1:
                        self.rowsum(i,rank)
                rank += 1

        #so now we have stuff above this line with xs in and stuff below with no xs
        #to keep the rows linearly independent we therefore have the remaining rows forming
        #a linearly indep set of Z guys
        return 2**(-((self.n-self.k) + rank))
