#(Non-calibrated) Analytical Heston model priced ATM capturing the Volatility Smile
#by K.TOmov

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# global Heston Parameters
S0 = 100.0 
T = 1.0
r = 0.02
N = 252
M = 1000
kappa = 3
theta = 0.20**2
v0 = 0.25**2
sigma = 0.6

# BSM
def bs_price(S, K, T, r, sigma, choice='c'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if choice == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif choice == 'p':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Choose either 'c' or 'p'")

# Implied Vol
def implied_vol(market_price, S, K, T, r, flag='c'):
    iv = np.zeros_like(market_price, dtype=float)
    for i, price in enumerate(market_price):
        try:
            f = lambda vol: bs_price(S, K[i], T, r, vol, flag) - price
            iv[i] = brentq(f, 1e-6, 5)
        except Exception:
            iv[i] = np.nan
    return iv

# Heston Simulation GBM
def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M, r):
    dt = T / N
    mu = [0, 0]
    cov = [[1, rho], [rho, 1]]
    S = np.full((N+1, M), S0)
    v = np.full((N+1, M), v0)
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    for i in range(1, N+1):
        S_prev, v_prev = S[i-1], v[i-1]
        S[i] = S_prev * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z[i-1,:,0])
        v[i] = np.maximum(v_prev + kappa * (theta - v_prev) * dt +
                          sigma * np.sqrt(v_prev * dt) * Z[i-1,:,1], 0)
    return S, v

# Two GBM correlation simulation
rho_p, rho_n = 0.98, -0.98
S_p, v_p = heston_model_sim(S0, v0, rho_p, kappa, theta, sigma, T, N, M, r)
S_n, v_n = heston_model_sim(S0, v0, rho_n, kappa, theta, sigma, T, N, M, r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
time = np.linspace(0, T, N+1)
ax1.plot(time, S_p)
ax1.set_title('Heston Model Asset Prices')
ax1.set_xlabel('T')
ax1.set_ylabel('Asset Price')

ax2.plot(time, v_p)
ax2.set_title('Heston Model Variance')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')
plt.tight_layout()
plt.show()

gbm = S0 * np.exp((r - theta/2) * T + np.sqrt(theta) * np.sqrt(T) * np.random.normal(size=M))

# Density comparison
plt.figure(figsize=(8,5))
sns.kdeplot(S_p[-1], label=r"Rho")
sns.kdeplot(S_n[-1], label=r"Rho")
sns.kdeplot(gbm, label="GBM")
plt.title('Asset Price Density under Heston vs. GBM')
plt.xlabel('TS')
plt.ylabel('Density')
plt.xlim([30, 150])
plt.legend()
plt.show()

# vol smile output
rho = -0.7
S, v = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M, r)

#Monte-Carlo option price
K = np.arange(20, 180, 2)
puts = np.exp(-r*T) * np.mean(np.maximum(K[:, None] - S[-1], 0), axis=1)
calls = np.exp(-r*T) * np.mean(np.maximum(S[-1] - K[:, None], 0), axis=1)

atm_idx = np.argmin(np.abs(K - S0))
print(f"Heston Model ATM Call Price (k={K[atm_idx]}): {calls[atm_idx]:.4f}")
print(f"Heston Model ATM Put Price  (k={K[atm_idx]}): {puts[atm_idx]:.4f}")

put_ivs = implied_vol(puts, S0, K, T, r, flag='p')
call_ivs = implied_vol(calls, S0, K, T, r, flag='c')

plt.figure(figsize=(8,5))
plt.plot(K, call_ivs, label='Call implied vol')
plt.plot(K, put_ivs, label='Put implied vol')
plt.title('Implied Volatility Smile from Heston Model')
plt.xlabel('Strike K')
plt.ylabel('IV')
plt.legend()
plt.show()

