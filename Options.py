import numpy as np

# Payoff of american put option
def payoff(S,K):
    return(np.maximum(K-S,0))
    
    
def GBM(So,dt,mu,sig,n,m):
    
    # get (n+1 x m/2) iid standard normal random samples
    Z=np.random.standard_normal((n+1,m))
    
    #Transform to simulate stoc prices in BS model
    s=So*np.exp(np.cumsum((mu-0.5*sig**2)*dt+sig*np.sqrt(dt)*Z, axis=0))
    return(s)

    
def GBManti(So,dt,mu,sig,n,m):
    # get (n+1 x m/2) iid standard normal random samples
    Z=np.random.standard_normal((n+1,int(m/2)))
    
    # s1 is stock prices corresponding to Z
    s1=So*np.exp(np.cumsum((mu-0.5*sig**2)*dt+sig*np.sqrt(dt)*Z, axis=0))
    
    # s2 is stock prices corresponding to -Z
    s2=So*np.exp(np.cumsum((mu-0.5*sig**2)*dt+sig*np.sqrt(dt)*(-1*Z), axis=0))
    return(s1,s2)
    

def GBMmm(So,dt,mu,sig,n,m):
    Z1=np.random.standard_normal((n+1,m))
    
    # Obtain Sample mean and variance of Z
    Zbar=np.mean(Z1, axis=1)
    sigZ=np.std(Z1, axis=1)
    
    ZbarM=np.ones((n+1,m))
    sigZM=np.ones((n+1,m))
    
    for i in range(0,n+1):
        ZbarM[i,:]=np.ones(m)*Zbar[i]
        sigZM[i,:]=np.ones(m)*sigZ[i]
    
    #transformed Z
    Z=(Z1-ZbarM)/sigZM
    
    # get corresponding stock prices
    s=So*np.exp(np.cumsum((mu-0.5*sig**2)*dt+sig*np.sqrt(dt)*(Z+ZbarM), axis=0))
    return(s)
    
