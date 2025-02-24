
####DYNAMIC DELTA-GAMMA BS HEDGING MODEL ######
#               by K.Tomov

import warnings
import numpy as np
from scipy.stats import norm
warnings.filterwarnings("ignore")

###FORMULA###

# Black-Scholes call option price
def call_option_price(initial_price, strike_price, T, rf, sigma):
    d1 = (np.log(initial_price / strike_price) + (rf + pow(sigma, 2) / 2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)
    return initial_price * norm.cdf(d1) - strike_price * np.exp(-rf * T) * norm.cdf(d2)

# Black-Scholes put option price
def put_option_price(initial_price, strike_price, T, rf, sigma):
    d1 = (np.log(initial_price / strike_price) + (rf + pow(sigma, 2) / 2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)
    return -initial_price * norm.cdf(-d1) + strike_price * np.exp(-rf * T) * norm.cdf(-d2)

# Delta of a European call option
def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# Gamma of a European option
def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


#HEDGING METHOD###

# Delta hedging
def delta_hedge(S0, r, T, n, N, mu, sigma, K, option_type="call"):
    option_params = (S0, K, T, r, sigma)
    n_steps = 2**n
    if option_type == "call":
        option_price = call_option_price(*option_params)
    else:
        option_price = put_option_price(*option_params)

    time_steps = T / n_steps
    time_discretisation = np.linspace(0, T, n_steps + 1).reshape(-1, 1)
    dW = np.zeros([n_steps + 1, N])
    dW[1 : n_steps + 1, :] = np.random.normal(0, 1, size=(n_steps, N)) * np.sqrt(time_steps)

    W = np.cumsum(dW, axis=0)
    time_grid = time_discretisation.reshape(-1, 1) * np.ones((n_steps + 1, N))
    increment = sigma * W + (mu - 0.5 * (sigma**2)) * time_grid
    St = S0 * np.exp(increment)  # Corrected: Removed -r as it's not part of GBM
    time_differences = (T - time_discretisation) * np.ones((n_steps + 1, N))
    discount_factor_matrix = np.exp(-r * time_grid)

    d1 = (
        np.log(St[1 : n_steps + 1, :] / (K * np.exp(-r * time_differences[1 : n_steps + 1, :])))
        + 0.5 * (r + sigma**2) * time_differences[1 : n_steps + 1, :]
    ) / (sigma * np.sqrt(time_differences[1 : n_steps + 1, :]))
    delta = norm.cdf(d1)
    price_difference = (
        St[1 : n_steps + 1, :] * discount_factor_matrix[1 : n_steps + 1, :]
        - St[:n_steps, :] * discount_factor_matrix[:n_steps, :]
    )
    sum_holding = np.sum(delta * price_difference, axis=0)
    X = np.exp(-r * T) * (option_price - sum_holding)

    upayoff = St[n_steps, :] - K
    payoff = upayoff * (upayoff > 0)
    PNL = X - payoff
    return PNL, X

# Gamma hedging and P&L calculation
def gamma_hedging(S0, K, T, r, sigma, mu, dt, n_steps):
    S = np.zeros(n_steps + 1)
    S[0] = S0
    call_price = np.zeros(n_steps + 1)
    call_price[0] = call_option_price(S[0], K, T, r, sigma)
    delta = np.zeros(n_steps + 1)
    delta[0] = delta_call(S[0], K, T, r, sigma)
    gamma_val = np.zeros(n_steps + 1)
    gamma_val[0] = gamma(S[0], K, T, r, sigma)
    portfolio_value = np.zeros(n_steps + 1)
    portfolio_value[0] = call_price[0] - delta[0] * S[0]
    cash = np.zeros(n_steps + 1)
    cash[0] = portfolio_value[0]
    PnL = np.zeros(n_steps + 1)

    # GBM stock price path and perform gamma hedging
    for i in range(1, n_steps + 1):
        Z = np.random.normal(0, 1)
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Update option price, delta, and gamma
        call_price[i] = call_option_price(S[i], K, T - i * dt, r, sigma)
        delta[i] = delta_call(S[i], K, T - i * dt, r, sigma)
        gamma_val[i] = gamma(S[i], K, T - i * dt, r, sigma)
        
        # Rebalance portfolio
        portfolio_value[i] = call_price[i] - delta[i] * S[i]
        cash[i] = cash[i - 1] * np.exp(r * dt) - (delta[i] - delta[i - 1]) * S[i]
        PnL[i] = portfolio_value[i] - portfolio_value[i - 1] + cash[i] - cash[i - 1]

    return S, call_price, delta, gamma_val, portfolio_value, cash, PnL

# Parameters

S0 = 70  # Initial stock price
K = 100   # Strike price
T = 1     # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = 0.13  # Volatility
mu = 0.05  # Expected return of the stock
dt = 1 / 252  # Time step (FOR GBM)
n_steps = int(T / dt)  # Number of steps

# Main execution
if __name__ == "__main__":
    # Delta hedging
    profit, holding = delta_hedge(
        S0=100, r=0.05, T=1, n=5, N=100, mu=0.02, sigma=0.2, K=100
    )
    print(
        f"Profit output of {profit.mean()} & portfolio value of {holding.mean()}"
    )

    # Gamma hedging
    S, call_price, delta, gamma_val, portfolio_value, cash, PnL = gamma_hedging(
        S0, K, T, r, sigma, mu, dt, n_steps
    )
    print("\nGamma Hedging Extended Results:")
    print(f"Stock Price of {S[-1]}")
    print(f"Call option price of {call_price[-1]}")
    print(f"Delta of {delta[-1]}")
    print(f"Gamma of {gamma_val[-1]}")
    print(f"Portfolio Value of {portfolio_value[-1]}")
    print(f"Realized P&L of {PnL[-1]}") 