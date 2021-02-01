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

def epsPrime(p, deltaTarg, eta, s, LPlus, m,precision=1e-15):
    #want to return eps such that
    #deltaPrime(p, eps, eta, s, LMin(deltaTarg, eta)+ L, m) = deltaTarg
    #print("deltaTarg", deltaTarg)
    #a = (np.power(np.sqrt((2/s)*(2+np.log(2/deltaTarg)))*(1+m) + np.sqrt(p),2) - p)/eta
    #b = p*np.sqrt(np.log(1/deltaTarg)/L)/(1-eta - eta*np.sqrt(np.log(1/deltaTarg)/L))
    #print("a", a, deltaPrime(p, a, eta, s, L+LMin(deltaTarg, eta), m))
    #print("b", b, deltaPrime(p, b, eta, s, L+LMin(deltaTarg, eta), m))
    #deltaPrime = 2*e^2 *exp(-s*(sqrt(p + eta*a) - sqrt(p))^2/(2*(m + 1)^2))
    #deltaPrime = exp(-L*((1-eta)*b / (p+eta*b)))
    #deltaPrime(p, a, eta, s, L, m) and deltaPrime(p, b, eta, s, L, m)  > deltaTarg
    #so 0, a, b < epsPrime

    #can we come up with some decent upper bound? - yes!
    #c and d are not individually upper bounds but their max is
    #c = (np.power(np.sqrt((2/s)*(2+np.log(1/deltaTarg)))*(1+m) + np.sqrt(p),2) - p)/eta
    #d = p*np.sqrt(np.log(0.5/deltaTarg)/L)/(1-eta - eta*np.sqrt(np.log(0.5/deltaTarg)/L))

    #epsLower = max(a,b)    
    #epsUpper = max(c,d)
    epsLower = 0
    epsUpper = 1
    while deltaPrime(p, epsUpper, eta, s, LPlus + LMin(deltaTarg, eta), m) > deltaTarg:
        epsLower = epsUpper
        epsUpper *= 2
    
    #so now the correct eps is between epsLower and epsUpper
    #just do midpoint division until we hit the required precision
    epsMid = (epsUpper + epsLower)/2
    while (epsUpper - epsLower)/epsMid > precision:
        #print(epsLower, epsUpper)
        epsMid = (epsUpper + epsLower)/2
        d = deltaPrime(p, epsMid, eta, s, LPlus + LMin(deltaTarg, eta), m)
        if d > deltaTarg:
            epsLower = epsMid
        else:
            epsUpper = epsMid    
    
    return epsMid


def eps2(p, deltaTarg, eta, s, tau, m, t, r,precision):
    LPlus = (tau - s*t*t*(t-r))/(s*r*r*r) - LMin(deltaTarg, eta)
    if LPlus < 0:
        print(p, deltaTarg, eta, s, tau, m, t, r)
    #print(s*t*t*(t-r) + s*L*r*r*r)
    return epsPrime(p, deltaTarg, eta, s, LPlus, m, precision)
    


def dDeltaPrimeDs(p, deltaTarg, s, tau, m, t, r, eta):
    """
    the derivative of eps2 with respect to s
    once we have substituted in L = tau/s
    """
    L = (tau - s*t*t*(t-r))/(s*r*r*r) - LMin(deltaTarg, eta)
    eps = epsPrime(p, deltaTarg, eta, s, L, m, precision=1e-15)
    
    a = -np.power(np.sqrt(p + eta*eps) - np.sqrt(p), 2)/(2*np.power(m + 1, 2))    
    b = -np.power((1-eta)*eps/(p+eta*eps), 2)

    return 2*np.exp(2)*a*np.exp(a*s) - np.exp(b*L)*b*(tau/(np.power(s,2))/r*r*r)
    
def dDeltaPrimeDsPositive(p, deltaTarg, s, tau, m, t, r, eta):
    L = (tau - s*t*t*(t-r))/(s*r*r*r) - LMin(deltaTarg, eta)
    eps = epsPrime(p, deltaTarg, eta, s, L, m, precision=1e-15)
    
    a = -np.power(np.sqrt(p + eta*eps) - np.sqrt(p), 2)/(2*np.power(m + 1, 2))    
    b = -np.power((1-eta)*eps/(p+eta*eps), 2)

    return ((np.log(-b*tau) - 2*np.log(float(s)) - 3*np.log(float(r)) + b*L) > (np.log(-a)+2+np.log(2.) + a*s))
    
    
def eps_at_particular_eta(p, deltaTarg, tau, m, eta, t, r, precision):
    """
    we evaluate the optimal epsilon for a fixed eta (varying s) 
    using the fact that the partial derivative of delta prime with respect to epsilon is non-negative
    by using the chain rule we can show that the derivative of eps2 with repsect to s has the same sign as the derivative of deltaPrime with respect to s
    """

    #here we have all the parameters chosen except s

    #with a fixed eta there is a range of s values we're allowed
    #by the equation s L  = s (L^+ + L_min(eta,delta)) = tau
    #always the minimum s is 1
    #the maximum s is floor(tau / ceil(L_min(eta,delta)))
    s_min = 1
    s_max = int(np.floor(tau / ( t*t*(t-r) + r*r*r*LMin(deltaTarg, eta))))
    #print(s_min, s_max)
    if s_min > s_max:
        #print(p, deltaTarg, tau, m, eta, s_min, s_max)
        return None
    if s_min == s_max:
        #print("hello")
        return eps2(p, deltaTarg, eta, s_min, tau, m, t, r, precision), int(np.ceil(s_min))

    #otherwise we have a range of s values to explore
    #we know that eps can have at most one fixed point with respect to s

    #possibilities
    # * we have a fixed point, it is a maximima, one of the end points is the optimum
    # * we have a fixed point, it is a point of inflexion, one of the end points is the optimum
    # * we have a fixed point, it is a minima and it is the optimum
    # * we have no fixed point, one of the end points is the optimum

    #step 1. we want to work out if eps is increasing or decreasing at the end points

    s_min_eps_value = eps2(p, deltaTarg, eta, s_min, tau, m, t,r,precision)
    s_max_eps_value = eps2(p, deltaTarg, eta, s_max, tau, m, t,r,precision)
    
    decreasing_at_start = (not dDeltaPrimeDsPositive(p, deltaTarg, s_min, tau, m, t,r,eta))
    decreasing_at_end = (not dDeltaPrimeDsPositive(p, deltaTarg, s_max, tau, m, t,r,eta))

    #this covers the cases where eps is monotone increasing or monotone decreasing over the whole range
    if decreasing_at_start == decreasing_at_end:
        #print("a")
        #print(s_min_eps_value, s_min_adj_eps_value)
        #print(s_max_eps_value, s_max_adj_eps_value)
        
        if s_min_eps_value < s_max_eps_value:
            return s_min_eps_value, int(np.ceil(s_min))
        else:
            return s_max_eps_value, int(np.ceil(s_max))

    #otherwise the sign of the derivative of eps changes
    #there is either a local min or a local max somewhere in the range
    if (not decreasing_at_start) and decreasing_at_end:
        #print("b")
        #there is a local maximum and the optimum is still one of the end points
        if s_min_eps_value < s_max_eps_value:
            return s_min_eps_value, int(np.ceil(s_min))
        else:
            return s_max_eps_value, int(np.ceil(s_max))
        
    #now we know there is a local minimum somewhere
    width = s_max - s_min
    #s_min_eps_value = eps2(p, deltaTarg, eta, s_min, tau, m, precision)
    #s_max_eps_value = eps2(p, deltaTarg, eta, s_max, tau, m, precision)
    
    while width > 1.5:
        midpoint = s_min + width/2.
        deriv_at_midpoint_positive = dDeltaPrimeDsPositive(p, deltaTarg, midpoint, tau, m, t,r,eta)
        midpoint_eps_value = eps2(p, deltaTarg, eta, midpoint, tau, m, t,r,precision)
        #print(midpoint_eps_value, midpoint_adj_eps_value)
        if deriv_at_midpoint_positive:
            #at the midpoint eps is increasing 
            s_max = midpoint
            s_max_eps_value = midpoint_eps_value
        else:
            #at the midpoint eps is decreasing 
            s_min = midpoint
            s_min_eps_value = midpoint_eps_value
            
        width = s_max - s_min
        
    if s_min_eps_value < s_max_eps_value:
        return s_min_eps_value, int(np.ceil(s_min))
    else:
        return s_max_eps_value, int(np.ceil(s_max))
    
def epsStar(p, deltaTot, tau, m, t, r, eta_prec=1e-10, deriv_prec=0.1, eps_prec=1e-15):
    #start with an initial guess for eta in the middle of the interval
    eta = 0.5

    eta_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, eta, t, r, eps_prec)
    eta_adj_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, eta + deriv_prec, t,r,eps_prec)

    lower_eta_bound = None
    upper_eta_bound = None

    if eta_adj_eps < eta_eps:
        lower_eta_bound = eta
        found_upper_bound = False
        upper_eta_bound = lower_eta_bound
        while not found_upper_bound:
            lower_eta_bound = upper_eta_bound
            upper_eta_bound = (1+upper_eta_bound)/2.
            #print("upper_eta_bound = ", upper_eta_bound)
            upper_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, upper_eta_bound, t,r,eps_prec)
            upper_adj_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, upper_eta_bound - deriv_prec, t,r,eps_prec)
            #print("epss = ", upper_adj_eps, upper_eps)
            if upper_adj_eps < upper_eps:
                found_upper_bound = True
    else:
        upper_eta_bound = eta
        found_lower_bound = False
        lower_eta_bound = upper_eta_bound
        while not found_lower_bound:
            upper_eta_bound = lower_eta_bound
            lower_eta_bound = lower_eta_bound/2.
            lower_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, lower_eta_bound, t,r,eps_prec)
            lower_adj_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, lower_eta_bound + deriv_prec, t,r,eps_prec)
            #print(p, deltaTot, tau, m, lower_eta_bound, eps_prec, deriv_prec)
            #print(lower_eps, lower_adj_eps)
            if lower_eps > lower_adj_eps:
                found_lower_bound = True

    #now we have an upper and lower bound so we just keep diving until we're done
    #print("upper = ", upper_eta_bound, "lower = ", lower_eta_bound)
    width = upper_eta_bound - lower_eta_bound
    midpoint_eps, s = None, None
    while width > eta_prec:
        midpoint = lower_eta_bound + width/2.
        midpoint_eps, s = eps_at_particular_eta(p, deltaTot, tau, m, midpoint, t,r,eps_prec)
        midpoint_adj_eps, s2 = eps_at_particular_eta(p, deltaTot, tau, m, midpoint + width*deriv_prec, t,r,eps_prec)

        if midpoint_eps < midpoint_adj_eps:
            #the derivative is positive at the midpoint
            upper_eta_bound = midpoint
        else:
            lower_eta_bound = midpoint
        width = upper_eta_bound - lower_eta_bound

    LPlusBest = int(np.ceil((tau - s*t*t*(t-r))/(s*r*r*r) - LMin(deltaTot, eta)))
    #if LPlusBest < 0:
    #    print(eta, deltaTot, LMin(deltaTot, eta))
    return (midpoint_eps, midpoint, s, LPlusBest)



def optimize(epsTot, deltaTot, t, measured_qubits, r, log_v, m, CH, AG, seed=None, threads=10):
    if seed != None:
        random.seed(seed)
    
    pStar = 1
    exitCondition = False
    k = 0
    s = np.ceil(-2*np.power(m + 1, 2)*np.log(deltaTot/(2*np.exp(2)))/epsTot)
    L = 1
    tauZero = s*t*t*(t-r) + s*L*r*r*r
    
    #print(tauZero)
    while not exitCondition:
        k += 1
        dtime = time.monotonic()
        #if np.power(2,k)*tauZero > 2000:
        eStar, eta, s, LPlus = epsStar(pStar, 6*deltaTot/np.power(np.pi*k,2), np.power(2,k)*tauZero, m,t,r)
        #print(eStar, eta,s,LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta), flush=True)
        #print(6*deltaTot/np.power(np.pi*k,2), deltaPrime(pStar, eStar, eta, s, LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta), m))
        #else:
        #    LPlus = 20
        #    eta = 0.5
        #    s = int(np.ceil(np.power(2,k)*tauZero/(LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta))))
        #    eStar = epsPrime(pStar, 6*deltaTot/np.power(np.pi*k,2), eta, s, LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta), m)
        optimize_time = time.monotonic() - dtime
        
        if eStar < epsTot:
            exitCondition = True

        
        totalL = int(np.ceil(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta)))
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

def optimize_with_phases(epsTot, deltaTot, t, measured_qubits, r, log_v, m, CH, AG, phases, seed=None, threads=10):
    if seed != None:
        random.seed(seed)
    
    pStar = 1
    exitCondition = False
    k = 0
    s = np.ceil(-2*np.power(m + 1, 2)*np.log(deltaTot/(2*np.exp(2)))/epsTot)
    L = 1
    tauZero = s*t*t*(t-r) + s*L*r*r*r
    K = 0
    #print(tauZero)
    start = time.monotonic()
    while not exitCondition:
        k += 1
        dtime = time.monotonic()
        #if np.power(2,k)*tauZero > 2000:
        #print("cost = ", np.power(2,k)*tauZero)
        eStar, eta, s, LPlus = epsStar(pStar, 6*deltaTot/np.power(np.pi*k,2), np.power(2,k)*tauZero, m,t,r)
        K += s*t*t*(t-r) + s*(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta))*r*r*r
        #print("actual cost = ", s*t*t*(t-r) + s*(LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta))*r*r*r)
        #print(6*deltaTot/np.power(np.pi*k,2), deltaPrime(pStar, eStar, eta, s, LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta), m))
        #else:
        #    LPlus = 20
        #    eta = 0.5
        #    s = int(np.ceil(np.power(2,k)*tauZero/(LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta))))
        #    eStar = epsPrime(pStar, 6*deltaTot/np.power(np.pi*k,2), eta, s, LPlus+LMin(6*deltaTot/np.power(np.pi*k,2), eta), m)
        optimize_time = time.monotonic() - dtime
        
        if eStar < epsTot:
            exitCondition = True

        
        totalL = int(np.ceil(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta)))
        #print(eta, s, totalL, LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta), measured_qubits, log_v, r)
        pHat = None
        if totalL >= threads:
            #def run_estimate_algorithm(seed):
            #    v1 = cPSCS.estimate_algorithm(int(round(s)), totalL, measured_qubits, log_v, r, seed, CH, AG).real
            #    return v1
            with Pool(threads) as p:
                seedvec = [random.randrange(1,10000) for _ in range(threads)]
                ans = p.starmap(cPSCS.estimate_algorithm_with_arbitrary_phases, [(int(round(s)), int(np.ceil(totalL/threads)), measured_qubits, log_v, r, seed, CH, AG, phases) for seed in seedvec] )
                mean_val = 0                
                for val in ans:
                    mean_val += val.real/threads
                pHat = mean_val
                
        else:
            pHat = cPSCS.estimate_algorithm_with_arbitrary_phases(int(round(s)), totalL, measured_qubits, log_v, r, random.randrange(1,10000), CH, AG, phases).real
        
        pStar = max(0, min(1, pStar, pHat + eStar))
        estimate_time = time.monotonic() - dtime

        print(k, pStar, pHat, eStar, epsTot, s, totalL, LPlus, eta, 6*deltaTot/np.power(np.pi*k,2), optimize_time, estimate_time)
    end = time.monotonic()
    return pStar, K, k, (start-end)


def hackedOptimise(p, m, epsTot, deltaTot, t, r, delta_UB, K_UUB):
    exitCondition = False
    #k = 0
    s = np.ceil(-2*np.power(m + 1, 2)*np.log(deltaTot/(2*np.exp(2)))/epsTot)
    L = 1
    tauZero = s*t*t*(t-r) + s*L*r*r*r

    k = 0
    K = 0
    pStar = 1
    #print(tauZero*np.power(2,k))
    #print(m, epsTot, deltaTot, t, r, delta_UB, K_UUB)
    #print("k pStar pHat eStar epsTot s totalL LPlus eta 6*deltaTot/np.power(np.pi*k2) LMin(6*deltaTot/np.power(np.pi*k2) eta) LMin(delta_UB/K_UUB eta_tilde)")
    while not exitCondition:
        k += 1
        eStar, eta, s, LPlus = epsStar(pStar, 6*deltaTot/np.power(np.pi*k,2), np.power(2,k)*tauZero, m,t,r)
        #print(p, eStar, eta,s,LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta))
        K += s*t*t*(t-r) + s*(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta))*r*r*r
        if eStar < epsTot:
            exitCondition = True
        totalL = int(round(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta)))
        #print(eta, s, totalL, LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta), measured_qubits, log_v, r)

        min_eps_prime = float("inf")
        eta_tilde_best = None
        for eta_tilde in np.linspace(1/100, 1-1/100, 100):
            if LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta) - LMin(delta_UB/K_UUB, eta_tilde) >= 1:
                val = epsPrime(p, delta_UB/K_UUB, eta, s, np.ceil(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta) - LMin(delta_UB/K_UUB, eta_tilde)), m, precision=1e-15)
                if val < min_eps_prime:
                    min_eps_prime = val
                    eta_tilde_best = eta_tilde
        pHat = p + min_eps_prime
        
        pStar = max(0, min(1, pStar, pHat + eStar))
        
        #print(k, pStar, pHat, eStar, epsTot, s, totalL, LPlus, eta, 6*deltaTot/np.power(np.pi*k,2), LMin(6*deltaTot/np.power(np.pi*k,2), eta), LMin(delta_UB/K_UUB, eta_tilde_best))
    #print(p, k)
    return p,(tauZero*(2**(k+1))), k



def clifT_eps1(del1,t,p,s):
    #computes the effective epsilon (eps1) error in |lambda-p| for the given
    #inputs.
    g=np.log2(4-2*np.sqrt(2));
    f1=2*np.power(2,(g*t/2)+p)
    f2=np.log(2*np.exp(2)/del1);
    return np.power((np.sqrt(f1*f2/s)+np.sqrt(p)),2)-p



def clifT_eps2(del2,L):
    #computes the effective epsilon tilde (eps2) reletive error |p_hat/lambda-1| for the given
    #inputs.
    return np.sqrt(-np.log(del2)/L)



def fake_child_alg(p,t,s,L):
    del_div=1e6#%this decides the granularity of the sampling of del1,del2 in [0,1]
    #%set probs for skewed/unskewed sampling of lambda and p_hat
    prob_eps1_over=0.5# %@0, always sets lambda=p-|eps1|; @1, always sets lambda=p+|eps1|
    prob_eps2_over=0.5# %@0, always sets p_hat=(1-|eps2|)lambda; @1, always sets p_hat=(1+|eps2|)lambda
    r= np.random(6)
    #%decides if lambda will be over or underestimate of p
    if r[5]<prob_eps1_over:
        g1=0
    else:
        g1=1
    K1=ceil(del_div*r(1)) #decides magnitude of lambda error from p
    K2=ceil(del_div*r(2)) #decides magnitude of p_hat error from lambda
    del1_p=K1/del_div
    del1_m=(K1-1)/del_div
    del2_p=K2/del_div
    del2_m=(K2-1)/del_div
    eps1_m=clifT_eps1(del1_m,t,secret_p,s)
    eps1_p=clifT_eps1(del1_p,t,secret_p,s)
    d1=eps1_p-eps1_m
    abs_eps1=eps1_m+d1*r(3)
    l=secret_p
    if g1 == 0:
        l += abs_eps1
    else:
        l -= abs_eps1
    eps2_m=clifT_eps2(del2_m,L)
    eps2_p=clifT_eps2(del2_p,L)
    d2=eps2_p-eps2_m
    abs_eps2=eps2_m+d2*r(4)
    return (1+np.power((-1), g2)*abs_eps2)*l
    

def fakeOptimise(p, m, epsTot, deltaTot, t, r, delta_UB, K_UUB):
    exitCondition = False
    #k = 0
    s = np.ceil(-2*np.power(m + 1, 2)*np.log(deltaTot/(2*np.exp(2)))/epsTot)
    L = 1
    tauZero = s*t*t*(t-r) + s*L*r*r*r

    k = 0
    K = 0
    pStar = 1
    #print(tauZero*np.power(2,k))
    #print(m, epsTot, deltaTot, t, r, delta_UB, K_UUB)
    #print("k pStar pHat eStar epsTot s totalL LPlus eta 6*deltaTot/np.power(np.pi*k2) LMin(6*deltaTot/np.power(np.pi*k2) eta) LMin(delta_UB/K_UUB eta_tilde)")
    
    while not exitCondition:
        k += 1
        eStar, eta, s, LPlus = epsStar(pStar, 6*deltaTot/np.power(np.pi*k,2), np.power(2,k)*tauZero, m,t,r)
        #print(p, eStar, eta,s,LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta))
        K += s*t*t*(t-r) + s*(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta))*r*r*r
        if eStar < epsTot:
            exitCondition = True
        totalL = int(round(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta)))
        #print(eta, s, totalL, LPlus, LMin(6*deltaTot/np.power(np.pi*k,2), eta), measured_qubits, log_v, r)

        min_eps_prime = float("inf")
        eta_tilde_best = None
        for eta_tilde in np.linspace(1/100, 1-1/100, 100):
            if LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta) - LMin(delta_UB/K_UUB, eta_tilde) >= 1:
                val = epsPrime(p, delta_UB/K_UUB, eta, s, np.ceil(LPlus + LMin(6*deltaTot/np.power(np.pi*k,2), eta) - LMin(delta_UB/K_UUB, eta_tilde)), m, precision=1e-15)
                if val < min_eps_prime:
                    min_eps_prime = val
                    eta_tilde_best = eta_tilde
        pHat = p + min_eps_prime
        
        pStar = max(0, min(1, pStar, pHat + eStar))

        #print(k, pStar, pHat, eStar, epsTot, s, totalL, LPlus, eta, 6*deltaTot/np.power(np.pi*k,2), LMin(6*deltaTot/np.power(np.pi*k,2), eta), LMin(delta_UB/K_UUB, eta_tilde_best))
    #print(p, k)
    return p,(tauZero*(2**(k+1))), k

    

    
if __name__ == "__main__":
    beta = np.log2(4. - 2.*np.sqrt(2.));
    t = 40
    m = np.power(2, beta*t/2)

    p = 0.0125
    eta = 0.5

    s = 1852065
    L = 2553
    
    #s = 100000
    #L = 1000
    # ps = np.linspace(0.01,0.99, 20, endpoint=True)
    # Ks = np.zeros_like(ps)
    
    # eps = 0.05
    # deltaTot = 1e-3

    # for i, p in enumerate(ps):
    #     p, K = hackedOptimise(p, m, eps, deltaTot)
    #     print(p, K)
    #     Ks[i] = K
    # plt.plot(ps, Ks)
    # plt.grid()
    # plt.show()

    p=0.5
    deltaTarg = 0.00000187631821559884
    tau = np.power(2,19)*411058688.0
    
    etas = np.linspace(0.01,0.99,100)
    epss = np.zeros_like(etas)
    for i, eta in enumerate(etas):
        eps, s = eps_at_particular_eta(p, deltaTarg, tau, 12.598212529123684, eta,t,8,1e-15)
        epss[i] = eps

    plt.plot(etas,epss)
    plt.grid()
    plt.show()
                          
