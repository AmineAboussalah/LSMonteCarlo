import numpy as np
from Options import GBM, payoff

def AmOptionV(S0,T,r,sigma,K,N,M):
    dt=T/N # time step length

    S=GBM(S0,dt,r,sigma,N,M)
    V=np.zeros((N+1,M))
    V[-1]=payoff(S[-1],K)
    Vtemp=V[-1]

    for n in range(N-1, 0, -1):
        beta=np.polyfit(S[n],Vtemp*np.exp(-r*dt),5)
        Cont=np.polyval(beta,S[n])
        Vtemp=np.where(payoff(S[n],K)>Cont, payoff(S[n],K),Vtemp*np.exp(-r*dt))
        V[n,:]=Vtemp
    
    V[0,:]=Vtemp*np.exp(-r*dt) 
    
    return(V)

