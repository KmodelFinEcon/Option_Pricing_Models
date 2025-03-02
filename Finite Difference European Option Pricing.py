####Finite Difference European Option Pricing####
#           by Kaloi Tomov

import numpy as np
from scipy.stats import norm

# Global Parameters
S = 100         # Current stock price
K = 100         # Strike price
T = 1.0         # tau Time to maturity (years)
r = 0.0425      # Risk-free rate
sigma = 0.1     # Annual Volatility
Smax = 1.4 * S  # Maximum stock price
M = 100         # Stock price steps
N = 1000        # time steps
EPS = 1e-10


def black_scholes_merton(S, K, T, r, sigma, option_type="call"):
    S = np.where(S == 0, EPS, S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def explicit_scheme(Smax, K, T, r, sigma, M, N, option_type="call"):
    dt = T / N
    dS = Smax / M
    S_values = np.linspace(0, Smax, M + 1)
    
    grid = np.zeros((M + 1, N + 1))
    
    if option_type == "call":
        grid[:, -1] = np.maximum(S_values - K, 0)
    else:
        grid[:, -1] = np.maximum(K - S_values, 0)
    
    t_grid = np.linspace(0, T, N + 1)
    if option_type == "call":
        grid[0, :] = 0
        grid[-1, :] = Smax - K * np.exp(-r * t_grid)
    else:
        grid[0, :] = K * np.exp(-r * t_grid)
        grid[-1, :] = 0
    
    S_int = S_values[1:-1]
    a = 0.5 * dt * (sigma**2 * S_int**2 / dS**2 - r * S_int / dS)
    b = 1 - dt * (sigma**2 * S_int**2 / dS**2 + r)
    c = 0.5 * dt * (sigma**2 * S_int**2 / dS**2 + r * S_int / dS)
    
    for n in range(N - 1, -1, -1):
        grid[1:-1, n] = (a * grid[:-2, n + 1] +
                         b * grid[1:-1, n + 1] +
                         c * grid[2:, n + 1])
    return grid, S_values


def implicit_scheme(Smax, K, T, r, sigma, M, N, option_type="call"):
    dt = T / N
    dS = Smax / M
    S_values = np.linspace(0, Smax, M + 1)
    grid = np.zeros((M + 1, N + 1))

    if option_type == "call":
        grid[:, -1] = np.maximum(S_values - K, 0)
    else:
        grid[:, -1] = np.maximum(K - S_values, 0)
    
    t_grid = np.linspace(0, T, N + 1)
    if option_type == "call":
        grid[0, :] = 0
        grid[-1, :] = Smax - K * np.exp(-r * t_grid)
    else:
        grid[0, :] = K * np.exp(-r * t_grid)
        grid[-1, :] = 0
    
    S_int = S_values[1:-1]
    a_int = 0.5 * dt * (sigma**2 * S_int**2 / dS**2 - r * S_int / dS)
    b_int = -(1 + dt * (sigma**2 * S_int**2 / dS**2 + r))
    c_int = 0.5 * dt * (sigma**2 * S_int**2 / dS**2 + r * S_int / dS)
    
    for n in range(N - 1, -1, -1):
        d = -grid[1:-1, n + 1].copy()
        d[0] -= a_int[0] * grid[0, n]
        d[-1] -= c_int[-1] * grid[-1, n]
        grid[1:-1, n] = thomas_algorithm(a_int, b_int, c_int, d)
    return grid, S_values

def thomas_algorithm(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


if __name__ == "__main__":
    S_range = np.linspace(EPS, Smax, M + 1)
    analytical_call = black_scholes_merton(S_range, K, T, r, sigma, option_type="call")
    analytical_put = black_scholes_merton(S_range, K, T, r, sigma, option_type="put")
    
    # Explicit FD price
    explicit_call_grid, S_explicit = explicit_scheme(Smax, K, T, r, sigma, M, N, option_type="call")
    explicit_put_grid, _ = explicit_scheme(Smax, K, T, r, sigma, M, N, option_type="put")
    
    # Implicit FD price
    implicit_call_grid, S_implicit = implicit_scheme(Smax, K, T, r, sigma, M, N, option_type="call")
    implicit_put_grid, _ = implicit_scheme(Smax, K, T, r, sigma, M, N, option_type="put")
    
    explicit_call_error = np.mean(np.abs(explicit_call_grid[:, 0] - analytical_call))
    explicit_put_error = np.mean(np.abs(explicit_put_grid[:, 0] - analytical_put))
    implicit_call_error = np.mean(np.abs(implicit_call_grid[:, 0] - analytical_call))
    implicit_put_error = np.mean(np.abs(implicit_put_grid[:, 0] - analytical_put))
    
    print(f"(Explicit Call): {explicit_call_error:.6f}")
    print(f"(Explicit Put):  {explicit_put_error:.6f}")
    print(f"(Implicit Call): {implicit_call_error:.6f}")
    print(f"(Implicit Put):  {implicit_put_error:.6f}")