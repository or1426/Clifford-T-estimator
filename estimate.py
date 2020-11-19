#! /usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import cPSCS
import time
from multiprocessing import Pool
import random

def deltaPrime(p, epsTot, eta, s, L, m):
    a = -s * np.power(np.sqrt(p + eta*epsTot) - np.sqrt(p), 2)/(2*np.power(m + 1, 2))
    b = -L * np.power((1-eta)*epsTot/(p+eta*epsTot), 2)
    return 2*np.exp(2)*np.exp(a) + np.exp(b)

def LMin(delta, eta):
    #print(delta, eta)
          
    return np.ceil(-np.power(eta/(1-eta), 2)*np.log(delta))

def epsPrime(p, deltaTarg, eta, s, L, m,precision=0.01):
    #want to return eps such that
    #deltaPrime(p, eps, eta, s, LMin(deltaTarg, eta)+ L, m) = deltaTarg

    #we know eps = 0 will make a delta that is too big
    #we expect that eps = 1 will make a delta that is too small but can't guarantee this (I think)
    #keep doubling epsUpper until we find an upper bound
    epsLower = 0
    epsUpper = 1

    while deltaPrime(p, epsUpper, eta, s, L+LMin(deltaTarg, eta), m) > deltaTarg:
        epsLower = epsUpper
        epsUpper *= 2
        if epsUpper > 2**20:            
            #print("WARNING: epsPrime function could not find a sensible upper bound")
            return None
    
    #so now the correct eps is between epsLower and epsUpper
    #just do midpoint division until we hit the required precision
    epsMid = (epsUpper + epsLower)/2
    while (epsUpper - epsLower)/epsMid > precision:
        epsMid = (epsUpper + epsLower)/2
        d = deltaPrime(p, epsMid, eta, s, L+LMin(deltaTarg, eta), m)
        if d > deltaTarg:
            epsLower = epsMid
        else:
            epsUpper = epsMid    
    
    return epsMid

def epsStar(p, deltaTot, tau, m, eta_prec=1e-3, s_samples=1000):
    #first lets come up with some bounds for eta
    #need Lmin < tau

    # a = tau/(-np.log(deltaTot))
    # eta_max = max((-a + np.sqrt(a + a*(1-a)))/(1-a), (-a - np.sqrt(a + a*(1-a)))/(1-a))

    # #if your inputs were sensible then 0 < eta_max < 1
    # #if not maybe we should mention this

    # if not (0 < eta_max < 1):
    #     print(a)
    #     print("WARNING: epsStar computed an eta_max of {}".format(eta_max))
    #     return None
    eta_max = 1-eta_prec
    #print("epsStar - tau = ", tau)
    etas = np.arange(eta_prec, eta_max, eta_prec)
    epsMin = float("inf")
    sBest, LBest, etaBest = None, None, None
    for eta in etas:        
        ss = np.ceil(np.linspace(1, tau/(1 + LMin(deltaTot, eta)),  s_samples)) 
        epsMinForThisEta = float("inf")
        sBestForThisEta, LBestForThisEta = None, None
        epsPrev = float("inf")
        sPrev = None
        LPrev = None
        for s in ss:
            L = int(np.floor(tau/s - LMin(deltaTot, eta)))
            if L < 1:
                L = 1
            #print(s,L, s*L, tau)
            #s = np.floor(tau/(L + ))
            if L+LMin(deltaTot, eta) >= 20:
                s = int(round(s))
                eps = epsPrime(p, deltaTot, eta, s, L + LMin(deltaTot, eta), m, precision=1e-4)
                #if eps== None:
                #                #print(s,L, eta)
                if eps != None and eps < epsPrev:
                    epsPrev = eps
                    sPrev = s
                    LPrev = L
                else:
                    epsMinForThisEta = epsPrev
                    sBestForThisEta = sPrev
                    LBestForThisEta = LPrev
                    break
        if epsMinForThisEta < epsMin:
            epsMin = epsMinForThisEta
            sBest = sBestForThisEta
            LBest = LBestForThisEta
            etaBest = eta
    if etaBest == None:
        print(p, deltaTot, tau, m, eta_prec, s_samples)
    return (epsMin, etaBest, sBest, LBest)



def optimize(epsTot, deltaTot, tPrime, measured_qubits, r, log_v, m, CH, AG, seed=None, threads=10):
    if seed != None:
        random.seed(seed)
    
    pStar = 1
    exitCondition = False
    k = 0
    s = -2*np.power(np.sqrt(m) + 1, 2)*np.log(deltaTot/(2*np.exp(2)))/epsTot
    L = 1
    tauZero = s*L
    #print(tauZero)
    while not exitCondition:
        k += 1
        dtime = time.monotonic()
        #if np.power(2,k)*tauZero > 2000:
        eStar, eta, s, LPlus = epsStar(pStar, 6*deltaTot/np.power(np.pi*k,2), np.power(2,k)*tauZero, m)
        #else:
        #    LPlus = 20
        #    eta = 0.5
        #    s = int(np.ceil(np.power(2,k)*tauZero/(LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta))))
        #    eStar = epsPrime(pStar, 6*deltaTot/np.power(np.pi*k,2), eta, s, LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta), m)
        optimize_time = time.monotonic() - dtime
        
        if eStar < epsTot:
            exitCondition = True
        dtime = time.monotonic()
        totalL = int(round(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta)))
        #print(eta, s, totalL, LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta), measured_qubits, log_v, r)
        pHat = None
        if totalL >= 10:
            #def run_estimate_algorithm(seed):
            #    v1 = cPSCS.estimate_algorithm(int(round(s)), totalL, measured_qubits, log_v, r, seed, CH, AG).real
            #    return v1
            with Pool(threads) as p:
                seedvec = [random.randrange(1,10000) for _ in range(threads)]
                ans = p.starmap(cPSCS.estimate_algorithm, [(int(round(s)), int(np.ceil(totalL/threads)), measured_qubits, log_v, r, seed, CH, AG) for seed in seedvec] )
                mean_val = 0                
                for val in ans:
                    mean_val += val.real/threads
                pHat = mean_val
                
        else:
            pHat = cPSCS.estimate_algorithm(int(round(s)), totalL, measured_qubits, log_v, r, random.randrange(1,10000), CH, AG).real
        
        pStar = max(0, min(1, pStar, pHat + eStar))
        estimate_time = time.monotonic() - dtime
        print(k, pStar, pHat, eStar, epsTot, s, totalL, LPlus, eta, 6*deltaTot/np.power(np.pi*k,2), optimize_time, estimate_time)
    return pStar
    
    
    

    
if __name__ == "__main__":
    p = np.power(2.,-3)
    
    deltaTarg = 1.149200570612527e-07
    
    #23 0.1475560492266207 0.1250202067816151 0.022638320922851562 0.02 39981459 998 997 0.156 1.149200570612527e-07 1.8804871661122888 124.6749424149748

    eta = 0.156
    
    s = 39981459
    L = 998
    
    beta = np.log2(4. - 2.*np.sqrt(2.));
    t = 10
    m = np.power(2, beta*t/2)

    pHat = 0.1475560492266207
    
    epsP = epsPrime(pHat, deltaTarg, eta, s, L, m,precision=0.01)
    deltaP = deltaPrime(pHat, epsP, eta, s, L, m)
    print(epsP)
    print(deltaP)
