from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class AGState:
    N : int # number of qubits
    x : np.ndarray
    z : np.ndarray
    r : np.ndarray
    
    @classmethod
    def basis(cls, N:int = None, s=None) -> StabState:
        if N == None and s == None:
            return cls(1, np.eye(2,1,dtype=np.uint8), np.eye(2,1,-1,dtype=np.uint8), np.zeros(2, dtype=np.uint8))
        elif N != None and s == None:
            return cls(N, np.eye(2*N,N,dtype=np.uint8), np.eye(2*N,N,-N,dtype=np.uint8), np.zeros(2*N, dtype=np.uint8))
        elif N != None and s == None:
            N = len(s)
            p = np.zeros(2*N, dtype=np.uint8)
            p[N:] = s
            return cls(N, np.eye(2*N,N,dtype=np.uint8), np.eye(2*N,N,-N,dtype=np.uint8), p)
        else:
            return basis(s)

    def _rowToStr(row):
        return "".join(map(str,row))

    def tab(self):
        """
        pretty "to string" method for small qubit numbers
        prints blocks F G M gamma v s
        with headings to indicate which is which
        """
        
        s  = str(self.N) + " "
        qubitNumberStrLen = len(s)
        matrix_width = self.N
        half_matrix_width = self.N//2
        s = "N" + " "*(qubitNumberStrLen -1 + half_matrix_width) + "x" + " "*matrix_width + "z" + " "*(matrix_width-half_matrix_width) + "p" + "\n" + s
        
        for i, (xr, zr, rr) in enumerate(zip(self.x, self.z, self.r)):
            if i == self.N:
                s += "\n"
            if i != 0:
                s += " "*qubitNumberStrLen
            
            s += AGState._rowToStr(xr) + " " + AGState._rowToStr(zr) + " " + str(rr) + "\n"

        return s

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
                
    def stabs(self):
        s = ""
        for i in range(self.N, 2*self.N):
            s = s+self.row2Str(i) + "\n"
        return s


    def destabs(self):
        s = ""
        for i in range(0, self.N):
            s = s+self.row2Str(i) + "\n"
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


    def rowmult(self, h,i):
        #left multiply row i onto row h

        phase = ((self.x[h]*self.z[i] - self.x[i]*self.z[h]).sum() % np.uint8(4))
            
        self.r[h] = (self.r[h] + self.r[i] +  (phase / np.uint8(2))) % np.uint8(2)
        
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]
            
    def rowswap(self, h, i):
        self.x[[h,i]] = self.x[[i,h]]
        self.z[[h,i]] = self.z[[i,h]]
        self.r[[h,i]] = self.r[[i,h]]

    def gausStab(self):
        """
        Do Gaussian elimination on the stabiliser bit without changing the state represented
        """
        i = self.N # to skip the destabiliser bits of the  matrix
        for j in range(self.N):
            fnz = np.flatnonzero(self.x[i:,j])+i
            if len(fnz) > 0:
                k = fnz[0]
                if k != i:
                    self.rowswap(k,i)
                for m in range(self.N, 2*self.N):
                    if m != i and self.x[m,j] == 1:
                        self.rowsum(m, i)
                i = i+1

        for j in range(self.N):
            fnz = np.flatnonzero(self.z[i:,j])+i
            if len(fnz) > 0:
                k = fnz[0]
                if k != i:
                    self.rowswap(k,i)
                for m in range(self.N, 2*self.N):
                    if m != i and self.z[m,j] == 1:
                        self.rowsum(m, i)
                i = i+1










                
