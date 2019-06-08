import numpy as np
from Options import GBM, payoff, GBManti, GBMmm


def LSMsimple(S0,T,r,sigma,K,N,M):
    # time step size
    dt=T/N 
    
    # get stock prices
    S=GBM(S0,dt,r,sigma,N,M)
    
    # V is matrix of simulated option prices
    V=np.zeros((N+1,M))
    
    # Final row of V contains time T option value,
    # which is just the payoff of the option
    V[-1]=payoff(S[-1],K)
    
    # get option prices using backwards recursive LSM algorithm
    for n in range(N-1, 0, -1):
        # get beta: the coefficients of least-squares regression model with 
        # polynomial basis functions
        beta=np.polyfit(S[n],V[n+1]*np.exp(-r*dt),5)
        
        # Approximate value of option if we do not exercise using 
        # polynomial regression coefficents we just obtained
        Cont=np.polyval(beta,S[n])
        
        # Time n value of option
        V[n]=np.where(payoff(S[n],K)>Cont, payoff(S[n],K),V[n+1]*np.exp(-r*dt))
    
    # Discount time 1 option value to time 0
    V[0]=V[1]*np.exp(-r*dt) 
    
    # Monte Carlo estimate of option price at all discretized 
    # points in time
    Vmc=np.mean(V[0])
    
    # Standard error of Monte Carlo estiamte
    se=(1/np.sqrt(M))*np.std(V[0])
    
    # Upper and lower 95% confidence interval bounds for MC estiamte
    lower=Vmc-se*1.96
    upper=Vmc+se*1.96
    
    return(Vmc,se,lower,upper)
    

def LSManti(S0,T,r,sigma,K,N,M):
    dt=T/N # time step length

    # get antithetic variates stock prices
    S=GBManti(S0,dt,r,sigma,N,M)
    S=np.array(S)
    # V is matrix of simulated option prices
    V=np.zeros((2,N+1,int(M/2)))
    
    # Final row of V contains time T option value,
    # which is just the payoff of the option
    V[0,-1,:]=payoff(S[0,-1,:],K)
    V[1,-1,:]=payoff(S[1,-1,:],K)

    for n in range(N-1, 0, -1):
        # get beta: the coefficients of least-squares regression model with 
        # polynomial basis functions
        beta0=np.polyfit(S[0,n,:],V[0,n+1,:]*np.exp(-r*dt),5)
        beta1=np.polyfit(S[1,n,:],V[1,n+1,:]*np.exp(-r*dt),5)
        
        # Approximate value of option if we do not exercise using 
        # polynomial regression coefficents we just obtained
        Cont0=np.polyval(beta0,S[0,n,:])
        Cont1=np.polyval(beta1,S[1,n,:])
        
        # Time n value of option
        V[0,n,:]=np.where(payoff(S[0,n,:],K)>Cont0, payoff(S[0,n,:],K),V[0,n+1,:]*np.exp(-r*dt))
        V[1,n,:]=np.where(payoff(S[1,n,:],K)>Cont1, payoff(S[1,n,:],K),V[1,n+1,:]*np.exp(-r*dt))


    # Discount time 1 option value to time 0
    V[0,0,:]=V[0,1,:]*np.exp(-r*dt) 
    V[1,0,:]=V[1,1,:]*np.exp(-r*dt) 
    
    V0=V[0,0,:]
    V1=V[1,0,:]
    # antithetic MC estimate of V0
    y=(V0+V1)
    Vmc=(1/M)*np.sum(y)
    
    # estimate se of V0
    var=(1/(M-1))*np.sum((y-Vmc)**2)
    se=np.sqrt(1/M)*np.sqrt(var)
    
    # Upper and lower 95% confidence interval bounds for estiamte
    lower=Vmc-se*1.96
    upper=Vmc+se*1.96
    
    return(Vmc,se,lower,upper)

def LSMmm(S0,T,r,sigma,K,N,M,a,b):
    dt=T/N # time step length
    

    S=GBMmm(S0,dt,r,sigma,N,M)
    
    
    V=np.zeros((N+1,M))
    
    V[-1]=payoff(S[-1],K)

    for n in range(N-1, 0, -1):
        # get beta: the coefficients of least-squares regression model with 
        # polynomial basis functions
        beta=np.polyfit(S[n],V[n+1]*np.exp(-r*dt),5)
        
        # Approximate value of option if we do not exercise using 
        # polynomial regression coefficents we just obtained
        Cont=np.polyval(beta,S[n])
        
        # Time n value of option
        V[n]=np.where(payoff(S[n],K)>Cont, payoff(S[n],K),V[n+1]*np.exp(-r*dt))
    
    # Discount time 1 option value to time 0
    V[0]=V[1]*np.exp(-r*dt) 
    Vo=V[0]
    
    # moment matching estimate for V0
    Vmc=np.mean(Vo)
    
    # get batch means
    vbatch=np.zeros(a)
    for i in range(1,a+1):
        vbatch[i-1]=np.mean(Vo[b*(i-1):b*i])
    
    # get se estimate for V0 using se of batches
    var=(b/(a-1))*np.sum((vbatch - Vmc)**2)
    se=(1/np.sqrt(M))*np.sqrt(var)
    
    # Upper and lower 95% confidence interval bounds for estiamt
    lower=Vmc-se*1.96
    upper=Vmc+se*1.96
    
    return(Vmc,se,lower,upper)    
