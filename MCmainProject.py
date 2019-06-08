import numpy as np
import datetime
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf 
import pandas as pd
from LSM import LSMsimple, LSManti, LSMmm

np.random.seed(12345)
# start date for old data
dayHist=datetime.date(2018,10,23)

# time 0: when option was written
day0=datetime.date(2018,11,23)

# time T: when option expires
dayT=datetime.date(2019,2,22)

# symbol of underlying stock
sym='NFLX'

# daily market risk-free rate of return
# found via https://www.bloomberg.com/markets/rates-bonds/government-bonds/us
r=0.024/90

#%%
# get historical data to choose model parameters
dataHist = yf.download(sym,dayHist,day0)
#%%
# time to expiry in days
T=len(pd.bdate_range(day0, dayT))-2 



# get historical prices of stock
sHist=dataHist['Open']
# number of historical prices
tHist=len(sHist)
sHist=sHist.values.reshape((tHist,))

# daily log rate of return on asset
X=np.log(sHist[1:]/sHist[0])

# asset volatility estimate
sigma=np.std(X)

# time 0 stick price is last historical value
S0=sHist[-1]

# strike price for at-the-money option
K=S0


 # number of simulations
M=10000
# number of time steps=number of days to expriy
N=T

# Output parameter values
print('T:', '%3f'%T)
print('sigma:', '%2f'%sigma)
print('r:', '%2f'%r)
print('So:', '%2f'%S0)
print('K:', '%2f'%K)


# time step length
dt=T/N 

#%%
# simple LSM
mc,ste,LB,UB=LSMsimple(S0,T,r,sigma,K,N,M)

# LSM with anithetic variates 
mc2,ste2,LB2,UB2=LSManti(S0,T,r,sigma,K,N,M)


# LSM with moment-matching

# number of batches to get se
a=100
# number of simulations per batch to get se
b=100
mc3,ste3,UB3,LB3=LSMmm(S0,T,r,sigma,K,N,M, a, b)

#%%
# Print Results
print('LSM with M = 10000: \n', 'Estimate of Vo: ', '%2f'%mc,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB, ',', '%2f'%UB, ')','\n','\n' )

print('LSM using Antithetic Variates with M = 10000: \n', 'Estimate of Vo: ', '%2f'%mc2,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste2, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB2, ',', '%2f'%UB2, ')','\n','\n' )

print('LSM using Moment Matching with M = 10000: \n', 'Estimate of Vo: ', '%2f'%mc3,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste3, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB3, ',', '%2f'%UB3, ')'  )

#%%
Mi=10000
mcV=[]
seV=[]
mcV2=[]
seV2=[]
mcV3=[]
seV3=[]
for i in range(0,150):
    temp=LSMsimple(S0,T,r,sigma,K,N,Mi)
    mcV.append(temp[0])
    seV.append(temp[1])
    
    temp2=LSManti(S0,T,r,sigma,K,N,Mi)
    mcV2.append(temp2[0])
    seV2.append(temp2[1])
    
    temp3=LSMmm(S0,T,r,sigma,K,N,Mi,a,b)
    mcV3.append(temp3[0])
    seV3.append(temp3[1])
    
#%%
print('Using LSM over 150 iterations the average: \n', 'Estimate of Vo is: ', np.mean(mcV), '\n',
      'Approximate standard error of estimate is: ', np.mean(seV), '\n', '\n')

print('Using LSM with antithetic variates over 150 iterations the average: \n', 'Estimate of Vo is: ', np.mean(mcV2), 
      '\n','Approximate standard error of estimate is: ', np.mean(seV2), '\n', '\n')

print('Using LSM with moment matching over 150 iterations the average: \n', 'Estimate of Vo is: ', np.mean(mcV3),'\n', 
      'Approximate standard error of estimate is: ', np.mean(seV3), '\n', '\n')
#%%
fig=plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111) 
#st = fig.suptitle("Histograms over 50 LSM Estimates with M=10,000 Simulations",fontweight='bold')

sub1 = fig.add_subplot(321)
sub1.set_title('LSM $V_0$ Estimates', fontsize=10,fontweight='bold')
sub1.hist(mcV, edgecolor='black', bins=20, color='b', range=(50.75,52.75))
sub1.set_ylim([0, 40])
sub1.set_xlabel('$V_0^{MC}$')

sub2 = fig.add_subplot(322)
sub2.set_title('SE of LSM Estimates', fontsize=10,fontweight='bold')
sub2.hist(seV, edgecolor='black', bins=20, color='b')
sub2.set_xlabel('$SE^{MC}$')

sub3 = fig.add_subplot(323)
sub3.set_title('LSM w. Anithetics $V_0$ Estimates', fontsize=10,fontweight='bold')
sub3.hist(mcV2, edgecolor='black', bins=20, color='r', range=(50.75,52.75)) 
sub3.set_ylim([0, 40])
sub3.set_xlabel('$V_0^{anti}$')

sub4 = fig.add_subplot(324)
sub4.set_title('SE of LSM w. Anithetics Estimate', fontsize=10,fontweight='bold')
sub4.hist(seV2, edgecolor='black', bins=20, color='r')
sub4.set_xlabel('$SE^{anti}$')

sub5 = fig.add_subplot(325)
sub5.set_title('LSM w. MM $V_0$ Estimate', fontsize=10,fontweight='bold')
sub5.hist(mcV3, edgecolor='black', bins=20, color='g', range=(50.75,52.75))
sub5.set_ylim([0, 40])
sub5.set_xlabel('$V_0^{MM}$')

sub6 = fig.add_subplot(326)
sub6.set_title('SE of LSM w. MM $V_0$ Estimates', fontsize=10,fontweight='bold')
sub6.hist(seV3, edgecolor='black', bins=20, color='g')
sub6.set_xlabel('$SE^{MM}$')


ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
plt.tight_layout()
fig.show()
#%%
Mi=100000
ai=250
bi=400
mci,stei,LBi,UBi=LSMsimple(S0,T,r,sigma,K,N,Mi)

mc2i,ste2i,LB2i,UB2i=LSManti(S0,T,r,sigma,K,N,Mi)

mc3i,ste3i,LB3i,UB3i=LSMmm(S0,T,r,sigma,K,N,Mi, ai, bi)

#%%
print('LSM with M = 100000: \n', 'Estimate of Vo: ', '%2f'%mci,
      '\n', 'Approximate standard error of estimate: ', '%2f'%stei, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LBi, ',', '%2f'%UBi, ')','\n','\n' )

print('LSM using Antithetic Variates with M = 100000: \n', 'Estimate of Vo: ', '%2f'%mc2i,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste2i, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB2i, ',', '%2f'%UB2i, ')','\n','\n' )

print('LSM using Moment Matching with M = 100000: \n', 'Estimate of Vo: ', '%2f'%mc3i,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste3i, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB3i, ',', '%2f'%UB3i, ')'  )
#%%

Mii=1000000
aii=1000
bii=1000
mcii,steii,LBii,UBii=LSMsimple(S0,T,r,sigma,K,N,Mii)

mc2ii,ste2ii,LB2ii,UB2ii=LSManti(S0,T,r,sigma,K,N,Mii)

mc3ii,ste3ii,LB3ii,UB3ii=LSMmm(S0,T,r,sigma,K,N,Mii, aii, bii)

#%%
print('LSM with M = 1000000: \n', 'Estimate of Vo: ', '%2f'%mcii,
      '\n', 'Approximate standard error of estimate: ', '%2f'%steii, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LBii, ',', '%2f'%UBii, ')','\n','\n' )

print('LSM using Antithetic Variates with M = 1000000: \n', 'Estimate of Vo: ', '%2f'%mc2ii,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste2ii, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB2ii, ',', '%2f'%UB2ii, ')','\n','\n' )

print('LSM using Moment Matching with M = 1000000: \n', 'Estimate of Vo: ', '%2f'%mc3ii,
      '\n', 'Approximate standard error of estimate: ', '%2f'%ste3ii, '\n', 
      'Approximate 95 % for estimate of Vo: (', '%2f'%LB3ii, ',', '%2f'%UB3ii, ')'  )
#%%