
###BS model Exact vs MC visualized by steps####
#           by K.Tomov

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as npr
plt.style.use('fivethirtyeight')

#formula

def BS_Call_Exact(S, X, r, sigma, T, t):
    d1 = (np.log(S / X) + (r + ((sigma**2) / 2)) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    ST = S * norm.cdf(d1) - X * np.exp(-r * (T - t)) * norm.cdf(d2)
    return ST

def BS_Call_MC(S,X,r,sigma,T,t,I):
    
    data = np.zeros((I, 2))
    z = np.random.normal(0, 1, [1, I])
    #z = npr.standard_normal(I)
    ST = S*np.exp((T-t)*(r - 0.5*sigma**2)+sigma*np.sqrt(T-t)*z)
    data[:,1] = ST - X
    average = np.sum(np.amax(data, axis=1))/float(I)
    return np.exp(-r*(T-t))*average

# 1) Varying time t in one dimension (different strike prices)>>>>

t = np.arange(0.75, 1.0, 0.0001)

plt.figure(figsize=(10, 8))
sns.lineplot(x=t, y=BS_Call_Exact(100, 95, 0.06, 0.3, 1, t), label='X = 95')
sns.lineplot(x=t, y=BS_Call_Exact(100, 98, 0.06, 0.3, 1, t), label='X = 98')
sns.lineplot(x=t, y=BS_Call_Exact(100, 100, 0.06, 0.3, 1, t), label='X = 100')
sns.lineplot(x=t, y=BS_Call_Exact(100, 105, 0.06, 0.3, 1, t), label='X = 105')

plt.xlabel('t')
plt.ylabel('C')
plt.ylim(-0.5, 8)
plt.legend()
plt.show()

# 2) Varying underlying price S in one dimension
s = np.arange(95, 105, 0.01)

plt.figure(figsize=(10, 8))
sns.lineplot(x=s, y=BS_Call_Exact(s, 100, 0.06, 0.3, 1, 0.99999), label='X = 100')
plt.xlabel('S')
plt.ylabel('C(t=T)')
plt.legend()
plt.show()

# Monte-Carlo method

r=0.06
sigma = 0.3
T = 1.0
S0 = 100
I = 100000
M = 100

dt = T/M
S = np.zeros((M+1, I))
S[0] = S0
for t in range(1,M+1):
    S[t] = S[t-1]*np.exp((r -0.5*sigma**2)*dt +sigma*np.sqrt(dt)*npr.standard_normal(I))
    
plt.figure(figsize=(10, 8))
plt.hist(S[-1], bins=100)
plt.xlabel('S(T)')
plt.ylabel('frequency')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(S[:, :6])
plt.xlabel('t')
plt.ylabel('S(t)')
plt.show()

#PRICING METHODS COMPARED:

print('Monte Carlo:',BS_Call_MC(100,95,0.06,0.3,1,.999,100000))
print('Exact: ',BS_Call_Exact(100,95,0.06,0.3,1,.999))

print('Monte Carlo:',BS_Call_MC(100,95,0.06,0.3,1,.75,100000))
print('Exact: ',BS_Call_Exact(100,95,0.06,0.3,1,.75))


#Convergence Visualized

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Black-Scholes formula for European call option
def black_scholes_call(S, K, T, r, sigma, t):
    tau = T - t
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return call_price

# Monte Carlo simulation for European call option
def monte_carlo_call(S, K, T, r, sigma, t, I):
    z = np.random.normal(0, 1, I)
    ST = S * np.exp((r - 0.5 * sigma**2) * (T - t) + sigma * np.sqrt(T - t) * z)
    payoff = np.maximum(ST - K, 0)
    call_price = np.exp(-r * (T - t)) * np.mean(payoff)
    return call_price


# Convergence plot for Monte Carlo vs Black-Scholes (T-t = 0.25)
df = pd.DataFrame(columns=['Iter', 'BSc'])

for i in range(1, 100000, 500):
    mc_price = monte_carlo_call(100, 95, 1, 0.06, 0.3, 0.75, i)
    new_row = pd.DataFrame({'Iter': [i], 'BSc': [mc_price]})
    df = pd.concat([df, new_row], ignore_index=True)

plt.figure(figsize=(10, 8))
exact_price = black_scholes_call(100, 95, 1, 0.06, 0.3, 0.75)
plt.hlines(exact_price, xmin=0, xmax=100000, linestyle='dotted', colors='red', label='Exact')
plt.plot(df.set_index('Iter'), lw=1.5, label='Monte Carlo')

plt.title('S=100, X=95, T-t=0.25')
plt.xlabel('Iterations')
plt.ylabel('Call Option Price (C)')
plt.ylim(exact_price - 1, exact_price + 1)
plt.legend()
plt.show()

# Convergence plot for Monte Carlo vs Black-Scholes (T-t = 0.01)
df = pd.DataFrame(columns=['Iter', 'BSc'])

for i in range(1, 100000, 500):
    mc_price = monte_carlo_call(100, 95, 1, 0.06, 0.3, 0.99, i)
    new_row = pd.DataFrame({'Iter': [i], 'BSc': [mc_price]})
    df = pd.concat([df, new_row], ignore_index=True)

plt.figure(figsize=(10, 8))
exact_price = black_scholes_call(100, 95, 1, 0.06, 0.3, 0.99)
plt.hlines(exact_price, xmin=0, xmax=100000, linestyle='dotted', colors='red', label='Exact')
plt.plot(df.set_index('Iter'), lw=1.5, label='Monte Carlo')

plt.title('S=100, X=95, T-t=0.01')
plt.xlabel('Iterations')
plt.ylabel('Call Option Price (C)')
plt.ylim(exact_price - 1, exact_price + 1)
plt.legend()
plt.show()


