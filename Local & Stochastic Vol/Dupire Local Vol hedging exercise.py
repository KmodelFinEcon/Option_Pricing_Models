#### Dupire Local Vol hedging exercise w/ yfinance data ####
#               by K.tomov

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import SmoothBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

###########################################
# Black-Scholes functions & Greeks
###########################################
def bsm_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option price.
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def delta_call(S, K, T, r, sigma):
    """
    Call option delta.
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def gamma_call(S, K, T, r, sigma):
    """
    Call option gamma.
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def implied_volatility(S, K, T, r, market_price, option_type="call"):
    """
    Implied volatility via scalar minimization.
    """
    objective = lambda sigma: (bsm_price(S, K, T, r, sigma, option_type) - market_price)**2
    result = minimize_scalar(objective, bounds=(0.01, 3), method="bounded")
    return result.x if result.success else np.nan

def time_to_maturity(expiration_date, trade_date):
    exp = pd.to_datetime(expiration_date).tz_localize(None)
    trade = pd.to_datetime(trade_date).tz_localize(None)
    return (exp - trade).days / 365.0

###########################################
# Data Acquisition and Implied Vol Surface
###########################################
ticker = "AMZN"
asset = yf.Ticker(ticker)
hist = asset.history(period="1y")
last_stock_price = hist["Close"].iloc[-1]

# Get option chain for a chosen expiration date.
expirations = asset.options
# Choose a later expiration if available (adjust index as needed)
chosen_expiration = expirations[15] if len(expirations) > 15 else expirations[-1]
opt_chain = asset.option_chain(chosen_expiration)
calls = opt_chain.calls.dropna().copy()

# Compute time to maturity (T) and mid prices for each option.
calls["T"] = calls["lastTradeDate"].apply(lambda d: time_to_maturity(chosen_expiration, d))
calls["mid"] = (calls["bid"] + calls["ask"]) / 2

# Use a constant risk-free rate (adjust as needed)
r_bsm = 0.0485

# Compute implied volatilities (for calls)
calls["iv"] = calls.apply(lambda row: 
                           implied_volatility(last_stock_price, row["strike"], row["T"], r_bsm, row["mid"], "call"),
                           axis=1)

# Prepare data for surface interpolation.
x = np.array(calls["strike"])
y = np.array(calls["T"])
z = np.array(calls["iv"])

# Create a regular grid in strike and maturity.
strike_range = np.linspace(x.min(), x.max(), 100)
maturity_range = np.linspace(y.min(), y.max(), 100)
K_mesh, T_mesh = np.meshgrid(strike_range, maturity_range)

# Use a smooth bivariate spline for interpolation.
spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.1)
IV_surface = spline.ev(K_mesh.ravel(), T_mesh.ravel()).reshape(K_mesh.shape)
IV_surface = np.maximum(IV_surface, 1e-6)

# Plot the implied volatility surface.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(K_mesh, T_mesh, IV_surface, cmap="viridis", edgecolor="none")
ax.set_xlabel("Strike (K)")
ax.set_ylabel("Time to Maturity (T)")
ax.set_zlabel("Implied Volatility")
ax.set_title("Implied Volatility Surface")
plt.show()

###########################################
# Local Volatility via Dupire's Formula
###########################################
def calculate_local_volatility(S, K_mesh, T_mesh, IV_surface, r):
    """
    Compute local volatility using Dupire's formula.
    """
    # Compute the BSM call price surface using the IV surface.
    call_prices = bsm_price(S, K_mesh, T_mesh, r, IV_surface, option_type="call")
    
    # Compute partial derivatives with respect to T and K.
    dC_dT = np.gradient(call_prices, T_mesh[:, 0], axis=0, edge_order=2)
    dC_dK = np.gradient(call_prices, K_mesh[0, :], axis=1, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K_mesh[0, :], axis=1, edge_order=2)
    d2C_dK2 = np.maximum(d2C_dK2, 1e-6)  # enforce positivity

    local_vol = np.sqrt(np.maximum((dC_dT + r * K_mesh * dC_dK) / (0.5 * K_mesh**2 * d2C_dK2), 1e-6))
    return local_vol

local_vol = calculate_local_volatility(last_stock_price, K_mesh, T_mesh, IV_surface, r_bsm)
local_vol = gaussian_filter(local_vol, sigma=1)

# Plot the local volatility surface.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(K_mesh, T_mesh, local_vol, cmap="plasma", edgecolor="none")
ax.set_xlabel("Strike (K)")
ax.set_ylabel("Time to Maturity (T)")
ax.set_zlabel("Local Volatility")
ax.set_title("Local Volatility Surface (Dupire)")
plt.show()

###########################################
# Delta Hedging Simulation
###########################################
def delta_hedge_simulation(S0, K, T, r, sigma, n_steps, n_paths, option_type="call"):
    dt = T / n_steps
    S_paths = np.zeros((n_steps + 1, n_paths))
    option_values = np.zeros((n_steps + 1, n_paths))
    deltas = np.zeros((n_steps + 1, n_paths))
    cash = np.zeros((n_steps + 1, n_paths))
    
    # Initial conditions
    S_paths[0] = S0
    option_values[0] = bsm_price(S0, K, T, r, sigma, option_type)
    deltas[0] = delta_call(S0, K, T, r, sigma)
    cash[0] = option_values[0] - deltas[0] * S0
    
    # Time-stepping simulation.
    for i in range(1, n_steps + 1):
        Z = np.random.normal(size=n_paths)
        S_paths[i] = S_paths[i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        t = i * dt
        remaining_T = max(T - t, 1e-6)  # avoid zero
        option_values[i] = bsm_price(S_paths[i], K, remaining_T, r, sigma, option_type)
        deltas[i] = delta_call(S_paths[i], K, remaining_T, r, sigma)
        # Adjust cash for change in delta
        cash[i] = cash[i - 1] * np.exp(r * dt) - (deltas[i] - deltas[i - 1]) * S_paths[i]
    
    # Final portfolio value and P&L.
    portfolio = cash[-1] + deltas[-1] * S_paths[-1]
    if option_type == "call":
        payoff = np.maximum(S_paths[-1] - K, 0)
    else:
        payoff = np.maximum(K - S_paths[-1], 0)
    PnL = portfolio - payoff
    return S_paths, option_values, deltas, cash, portfolio, payoff, PnL

###########################################
# Gamma Hedging Simulation
###########################################
def gamma_hedge_simulation(S0, K, T, r, sigma, mu, n_steps):
    dt = T / n_steps
    S = np.zeros(n_steps + 1)
    option_vals = np.zeros(n_steps + 1)
    deltas = np.zeros(n_steps + 1)
    gammas = np.zeros(n_steps + 1)
    portfolio = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    
    # Initial conditions
    S[0] = S0
    option_vals[0] = bsm_price(S0, K, T, r, sigma, "call")
    deltas[0] = delta_call(S0, K, T, r, sigma)
    gammas[0] = gamma_call(S0, K, T, r, sigma)
    portfolio[0] = option_vals[0] - deltas[0] * S0
    cash[0] = portfolio[0]
    
    for i in range(1, n_steps + 1):
        Z = np.random.normal()
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        t = i * dt
        remaining_T = max(T - t, 1e-6)
        option_vals[i] = bsm_price(S[i], K, remaining_T, r, sigma, "call")
        deltas[i] = delta_call(S[i], K, remaining_T, r, sigma)
        gammas[i] = gamma_call(S[i], K, remaining_T, r, sigma)
        portfolio[i] = option_vals[i] - deltas[i] * S[i]
        cash[i] = cash[i - 1] * np.exp(r * dt) - (deltas[i] - deltas[i - 1]) * S[i]
    
    # Final P&L calculation.
    PnL = cash[-1] + deltas[-1] * S[-1] - option_vals[-1]
    return S, option_vals, deltas, gammas, portfolio, cash, PnL

###########################################
# Main Execution
###########################################
if __name__ == "__main__":
    # Delta Hedging Simulation Parameters
    S0 = 100
    K = 100
    T = 1  # 1 year
    r = 0.05
    sigma = 0.2
    n_steps = 32  # for example, 32 time steps
    n_paths = 1000  # number of simulation paths

    S_paths, option_vals, deltas, cash, portfolio, payoff, pnl = delta_hedge_simulation(
        S0, K, T, r, sigma, n_steps, n_paths, option_type="call"
    )
    print("Delta Hedging Simulation:")
    print(f"Mean PnL: {np.mean(pnl):.4f}  Std PnL: {np.std(pnl):.4f}")
    
    # Gamma Hedging Simulation Parameters
    mu = 0.23  # drift for simulation
    n_steps_gamma = 100

    S_gamma, option_vals_gamma, deltas_gamma, gammas_gamma, portfolio_gamma, cash_gamma, pnl_gamma = gamma_hedge_simulation(
        S0, K, T, r, sigma, mu, n_steps_gamma
    )
    print("\nGamma Hedging Simulation:")
    print(f"Final Stock Price: {S_gamma[-1]:.4f}")
    print(f"Final Option Price: {option_vals_gamma[-1]:.4f}")
    print(f"Final Delta: {deltas_gamma[-1]:.4f}")
    print(f"Final Gamma: {gammas_gamma[-1]:.4f}")
    print(f"Final Portfolio Value: {portfolio_gamma[-1]:.4f}")
    print(f"Realized P&L: {pnl_gamma:.4f}")
    
    
    