####Heston VoL surface BS Option pricing from Market Data & Delta/Gamma hedging####
#               by K.Tomov

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.stats import norm, ncx2
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import griddata
import scipy.stats as stats
import random

#Option Chain Data collection
MSFT = yf.Ticker('MSFT')
hist = MSFT.history(period="1y")
log_returns = np.log(hist["Close"].pct_change())
S = hist["Close"].iloc[-1]
expirations = MSFT.options

#expiration date setup
expiration_date = expirations[3] 
options = MSFT.option_chain(expiration_date)
call_op = options.calls.dropna().reset_index(drop=True)
K_list = call_op["strike"].values
trade_dates = call_op["lastTradeDate"].values

#TTM computation in years
def time_to_maturity(expiration, trade_day):
    exp = pd.to_datetime(expiration).tz_localize(None) 
    trade = pd.to_datetime(trade_day).tz_localize(None)  
    return (exp - trade).days / 365

# Compute times-to-maturity for each option
first_time = time_to_maturity(expiration_date, call_op['lastTradeDate'].iloc[0])
times = call_op['lastTradeDate'].apply(lambda d: time_to_maturity(expiration_date, d))
print("Time-to-maturity for options:\n", times)

# Monte-Carlo Simulation Parameters
T = 252 # Number of time steps > function of days
dt = 1/252 # Time step size
N = 50 #Number of simulation paths
random.seed(123)
np.random.seed(123)

#Inition Heston Parameters:

r = 0.042
theta = 0.02
kappa = 0.02
sigma = 0.07
rho = -0.5
v0 = 1.5

# ---------------------------
# --Heston Model Simulation--
# ---------------------------

init_params = [r, theta, kappa, sigma, rho, v0]

def heston_vol(kappa, theta, sigma, Z, T, dt, v_0):
    v = np.zeros((T, N))
    v[0, :] = v_0
    for t in range(1, T):
        v[t, :] = np.maximum(v[t-1, :] + kappa * (theta - v[t-1, :]) * dt +
                              sigma * np.sqrt(v[t-1, :] * dt) * Z[t-1, :], 1e-6)# this Ensure non-negativity using a floor (1e-6)
    return v

def stoch_price(S, r, v, Z, dt):
    S_paths = np.zeros((T, N))
    S_paths[0, :] = S
    for t in range(1, T):
        S_paths[t, :] = S_paths[t-1, :] + S_paths[t-1, :]*r*dt + np.sqrt(v[t-1, :]*dt)*S_paths[t-1, :]*Z[t-1, :]
    return S_paths

cov_matrix = [[dt, init_params[4]*dt],
              [init_params[4]*dt, dt]]

dZ = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=(T-1, N))
Z1 = dZ[:, :, 0]
Z2 = dZ[:, :, 1]

# Simulate volatility and stock price paths
v = heston_vol(init_params[2], init_params[1], init_params[3], Z2, T, dt, init_params[-1])
S_paths = stoch_price(S, init_params[0], v, Z1, dt)

# -------------------------------
# Maximum Likelihood Estimation-- 
# -------------------------------

def heston_log_likelihood(params, S_paths, v, dt, T):
    r, theta, kappa, sigma, rho, v_0 = params
    log_likelihood = 0.0
    for t in range(1, T):
        c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
        df = 4 * kappa * theta / sigma**2
        nc = c * v[t-1, :] * np.exp(-kappa * dt)
        p_v = ncx2.pdf(2 * c * v[t, :], df, nc) + 1e-10
        mu_q = S_paths[t-1, :] + (r - 0.5 * v[t-1, :]) * dt
        sigma_q = np.sqrt(v[t-1, :] * dt)
        p_q = norm.pdf(S_paths[t, :], loc=mu_q, scale=sigma_q) + 1e-10
        log_likelihood += np.sum(np.log(p_v)) + np.sum(np.log(p_q))
    return -log_likelihood  # Negative log-likelihood for minimization

bounds = [(-10, 10), (0, None), (0, None), (0, None), (-1, 1), (0, None)]
result = minimize(heston_log_likelihood, init_params, args=(S_paths, v, dt, T),
                  bounds=bounds, method="L-BFGS-B") #L-BFGS-B Optimization method used

print("Estimated parameters (MLE):", result.x)

# -------------------------
# --Heston Option Pricing--
# -------------------------

K = call_op["strike"].iloc[0]
v_t = v[-1, -1]  # last simulated variance value

# Updated Heston characteristic function

def heston_characteristic_function(tau, S, v, r, kappa, theta, sigma, rho, j):
    i = 1j #probability weighting
    if j == 1:
        u_val = 0.5
        b = kappa - rho * sigma
    elif j == 2:
        u_val = -0.5
        b = kappa
    else:
        raise ValueError("j must be 1 or 2")
    
    d = np.sqrt((rho * sigma * i * u_val - b)**2 + sigma**2 * (u_val**2 + i * u_val))
    g = (b - rho * sigma * i * u_val + d) / (b - rho * sigma * i * u_val - d)
    
    C = r * i * u_val * tau + (kappa * theta / sigma**2) * ((b - rho * sigma * i * u_val + d) * tau - 
         2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
    D = (b - rho * sigma * i * u_val + d) / sigma**2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    
    return np.exp(C + D * v + i * u_val * np.log(S))

def call_price(S, P1, P2, T, r, K):
    return S * P1 - K * np.exp(-r * T) * P2

def compute_Pj(j, K, S, v, r, kappa, theta, sigma, rho, tau):    # Integrand for probability Pj
    integrand = lambda u: np.real(np.exp(-1j * u * np.log(K)) *
                                  heston_characteristic_function(tau, S, v, r, kappa, theta, sigma, rho, j) / (1j * u))
    integral, _ = quad(integrand, 0, np.inf, limit=100)
    return 0.5 + (1 / np.pi) * integral

def optimal_values(params, S, K, T, last_price):# Objective for calibrating to market option price
    r, kappa, theta, sigma, rho, v = params
    P1 = compute_Pj(1, K, S, v, r, kappa, theta, sigma, rho, T)
    P2 = compute_Pj(2, K, S, v, r, kappa, theta, sigma, rho, T)
    model_price = call_price(S, P1, P2, T, r, K)
    return (model_price - last_price) ** 2

initial_pricing_params = [result.x[0], result.x[2], result.x[1], result.x[3], result.x[4], v_t]# Use MLE estimates to initialize pricing calibration and the last simulated variance for v
last_price_market = call_op["lastPrice"].iloc[0]

# L-BFGS-B method for calibration
pricing_bounds = [(-10, 10), (0, None), (0, None), (0, None), (-1, 1), (0, None)]
optimal_result = minimize(optimal_values, initial_pricing_params, args=(S, K, first_time, last_price_market),
                          bounds=pricing_bounds, method="L-BFGS-B")
print("Optimized pricing parameters:", optimal_result.x)

r_opt, kappa_opt, theta_opt, sigma_opt, rho_opt, v_opt = optimal_result.x
P1 = compute_Pj(1, K, S, v_opt, r_opt, kappa_opt, theta_opt, sigma_opt, rho_opt, first_time)
P2 = compute_Pj(2, K, S, v_opt, r_opt, kappa_opt, theta_opt, sigma_opt, rho_opt, first_time)
heston_price = call_price(S, P1, P2, first_time, r_opt, K)
print("\nHeston option price:", heston_price)

# Compute Heston prices for all strikes in the call options chain
call_op["Heston price"] = call_op.apply(lambda row: call_price(
    S,
    compute_Pj(1, row["strike"], S, v_opt, r_opt, kappa_opt, theta_opt, sigma_opt, rho_opt, time_to_maturity(expiration_date, row["lastTradeDate"])),
    compute_Pj(2, row["strike"], S, v_opt, r_opt, kappa_opt, theta_opt, sigma_opt, rho_opt, time_to_maturity(expiration_date, row["lastTradeDate"])),
    time_to_maturity(expiration_date, row["lastTradeDate"]),
    r_opt,
    row["strike"]
), axis=1)

print("\nCall options chain with Heston prices:\n", call_op)

# ------------------------------------------------
# --Black-Scholes and Implied Volatility Surface--
# ------------------------------------------------

N_cdf = norm.cdf
def black_scholes(S0, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * N_cdf(d1) - K * np.exp(-r * T) * N_cdf(d2)
    else:
        return K * np.exp(-r * T) * N_cdf(-d2) - S0 * N_cdf(-d1)

def implied_vol(S0, K, T, r, option_price, option_type="call"):
    payoff = lambda sigma: (black_scholes(S0, K, T, r, sigma, option_type) - option_price)**2
    result = minimize_scalar(payoff, bounds=(0.001, 3), method="bounded")
    return result.x if result.success else np.nan

# Calculate implied volatility for the first option price
impl_vol = implied_vol(S, K, first_time, r_opt, last_price_market, option_type="call")
print("\nImplied volatility (first option):", impl_vol)

# Compute implied volatilities for each option in the chain using the Heston price
call_op["Implied Vol"] = call_op.apply(lambda row: implied_vol(
    S, row["strike"], time_to_maturity(expiration_date, row["lastTradeDate"]), r_opt, row["Heston price"], "call"
), axis=1)

# ------------------------
# Implied Volatility Surface
# ------------------------

x = np.array(K_list)
y = np.array(times)
z = np.array(call_op["Implied Vol"])

# grid for strikes and maturities
K_grid = np.linspace(x.min(), x.max(), 50)
T_grid = np.linspace(y.min(), y.max(), 50)
K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

# Interpolate implied volatilities onto the grid
Z_mesh = griddata((x, y), z, (K_mesh, T_mesh), method='cubic')

# Heston Volatility surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, Z_mesh, cmap='viridis', edgecolor='none')
ax.set_xlabel("Strike (K)")
ax.set_ylabel("Time to Maturity")
ax.set_zlabel("Implied Volatility")
ax.set_title("Implied Volatility Surface")
plt.show()

#Delta and Gamma for hedging at different strikes and maturities

def black_scholes_delta(S0, K, T, r, sigma, option_type='call'):
    """
    Computes the Black-Scholes delta.
    For a call, delta = N(d1); for a put, delta = N(d1) - 1.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

delta_first_option = black_scholes_delta(S, K, first_time, r_opt, impl_vol, option_type='call')
print("\nBlack-Scholes delta for the first option:", delta_first_option)

# Compute delta for each option in the chain and add as a new column.
call_op["BS Delta"] = call_op.apply(lambda row: black_scholes_delta(
    S,
    row["strike"],
    time_to_maturity(expiration_date, row["lastTradeDate"]),
    r_opt,
    row["Implied Vol"],  # using the implied vol from Heston pricing calibration
    option_type='call'
), axis=1)

def black_scholes_gamma(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return gamma

gamma_first_option = black_scholes_gamma(S, K, first_time, r_opt, impl_vol)
print("\nBlack-Scholes gamma for the first option:", gamma_first_option)

# Compute gamma for each option in the chain and add as a new column.
call_op["BS Gamma"] = call_op.apply(lambda row: black_scholes_gamma(
    S,
    row["strike"],
    time_to_maturity(expiration_date, row["lastTradeDate"]),
    r_opt,
    row["Implied Vol"]  # using the Heston-implied volatility
), axis=1)

print("\nOptions chain with Black-Scholes sensitivities:\n", 
      call_op[['strike', 'lastPrice', 'Heston price', 'Implied Vol', 'BS Delta', 'BS Gamma']])
