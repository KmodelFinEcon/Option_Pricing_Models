#Enhanced LSMC American Option model with Dividend yield, drift/Jump diffusion term extension
#by K.tomov

from time import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Global Option Parameters

OPTION_TYPE = 'put'   # 'call' or 'put'
S0 = 40.0             # Initial stock price
STRIKE = 46.0         # Strike price
T = 1.0               # Time to maturity (years)
M = 100                # Number of time steps
R = 0.043              # Risk-free rate
DIV = 0.014            # Dividend yield (2%)
SIGMA = 0.5           # Volatility (calibrate from vol surface)
SIMULATIONS = 15000   # Number of simulation paths

# Extra drift and jump diffusion parameters

ME = 0.01       # Small extra drift adjustment
LAMBDAJ = 0.3     # Jump intensity (expected jumps per year)
MJ = -0.02       # Mean of the log jump size
SIGMAJ = 0.2      # Standard deviation of the log jump size

#objective function

def GBMwithjump(S0, T, M, r, div, sigma, simulations, mu_extra=0.0, lambda_jump=0.0, mu_jump=0.0, sigma_jump=0.0, seed=300):
    np.random.seed(seed)
    dt = T / M
    discount = np.exp(-r * dt)
    paths = np.zeros((M + 1, simulations))
    paths[0] = S0
    
    for t in range(1, M + 1):
        Z = np.random.standard_normal(simulations)
        dN = np.random.poisson(lambda_jump * dt, simulations)# Poisson jump: number of jumps occurring in dt
        jump_factor = np.exp(mu_jump + sigma_jump * np.random.randn(simulations)) - 1
        jump_component = dN * jump_factor
        drift = (r - div + mu_extra - 0.5 * sigma**2) * dt #drift and diffusion parts:
        diffusion = sigma * np.sqrt(dt) * Z
        paths[t] = paths[t - 1] * np.exp(drift + diffusion) * (1 + jump_component) #apply replicability

    return paths, dt, discount #paths: simulations array of simulated stock prices - dt: the time steps size - discount: discount factor as function of timestep

def compute_payoff(prices, strike, option_type): #payoff matrix

    if option_type == 'call':
        payoff = np.maximum(prices - strike, 0)
    elif option_type == 'put':
        payoff = np.maximum(strike - prices, 0)
    else:
        raise ValueError("choose between both")
    return payoff

def compute_value_vector(prices, payoff, discount, M): #backward induction

    value_matrix = np.zeros_like(payoff) #terminal payoff
    value_matrix[-1, :] = payoff[-1, :]
    for t in range(M - 1, 0, -1):
        itm = payoff[t, :] > 0
        if np.sum(itm) > 0:
            coeff = np.polyfit(prices[t, itm], value_matrix[t + 1, itm] * discount, 5)
            continuation = np.polyval(coeff, prices[t, :])
        else:
            continuation = value_matrix[t + 1, :] * discount
        value_matrix[t, :] = np.where(payoff[t, :] > continuation, payoff[t, :],value_matrix[t + 1, :] * discount)
    return value_matrix[1, :] * discount# Discount back one time step to get the current value estimate

def compute_price(option_type, S0, strike, T, M, r, div, sigma, simulations,mu_extra=0.0, lambda_jump=0.0, mu_jump=0.0, sigma_jump=0.0):

    prices, dt, discount = GBMwithjump(S0, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump,sigma_jump)
    payoff = compute_payoff(prices, strike, option_type)
    value_vector = compute_value_vector(prices, payoff, discount, M)
    return np.mean(value_vector)

# Finite Difference estimation of Greeks

def compute_delta(option_type, S0, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump):
    diff = S0 * 0.01
    price_up = compute_price(option_type, S0 + diff, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    price_down = compute_price(option_type, S0 - diff, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    return (price_up - price_down) / (2 * diff)

def compute_gamma(option_type, S0, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump):
    diff = S0 * 0.01
    delta_up = compute_delta(option_type, S0 + diff, strike, T, M, r, div, sigma,simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    delta_down = compute_delta(option_type, S0 - diff, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    return (delta_up - delta_down) / (2 * diff)

def compute_vega(option_type, S0, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump):
    diff = sigma * 0.01
    price_up = compute_price(option_type, S0, strike, T, M, r, div, sigma + diff, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    price_down = compute_price(option_type, S0, strike, T, M, r, div, sigma - diff, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    return (price_up - price_down) / (2 * diff)

def compute_rho(option_type, S0, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump):
    diff = r * 0.01
    if (r - diff) < 0:
        price_up = compute_price(option_type, S0, strike, T, M, r + diff, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
        price_down = compute_price(option_type, S0, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
        return (price_up - price_down) / diff
    else:
        price_up = compute_price(option_type, S0, strike, T, M, r + diff, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
        price_down = compute_price(option_type, S0, strike, T, M, r - diff, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
        return (price_up - price_down) / (2 * diff)

def compute_theta(option_type, S0, strike, T, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump):
    diff = 1 / 252.0  # one trading day
    price_up = compute_price(option_type, S0, strike, T + diff, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    price_down = compute_price(option_type, S0, strike, T - diff, M, r, div, sigma, simulations, mu_extra, lambda_jump, mu_jump, sigma_jump)
    return (price_down - price_up) / (2 * diff)

# Main execusion

if __name__ == '__main__':
    option_price = compute_price(OPTION_TYPE, S0, STRIKE, T, M, R, DIV, SIGMA, SIMULATIONS, ME, LAMBDAJ, MJ, SIGMAJ)
    delta_val = compute_delta(OPTION_TYPE, S0, STRIKE, T, M, R, DIV, SIGMA,SIMULATIONS, ME, LAMBDAJ, MJ, SIGMAJ)
    gamma_val = compute_gamma(OPTION_TYPE, S0, STRIKE, T, M, R, DIV, SIGMA,SIMULATIONS, ME, LAMBDAJ, MJ, SIGMAJ)
    vega_val = compute_vega(OPTION_TYPE, S0, STRIKE, T, M, R, DIV, SIGMA,SIMULATIONS, ME, LAMBDAJ, MJ, SIGMAJ)
    rho_val = compute_rho(OPTION_TYPE, S0, STRIKE, T, M, R, DIV, SIGMA,SIMULATIONS, ME, LAMBDAJ, MJ, SIGMAJ)
    theta_val = compute_theta(OPTION_TYPE, S0, STRIKE, T, M, R, DIV, SIGMA,SIMULATIONS, ME, LAMBDAJ, MJ, SIGMAJ)

    #greeks output

    print("Longstaff-Schwartz Monte-Carlo:", option_price)
    print("Delta:", delta_val)
    print("Gamma:", gamma_val)
    print("Vega:", vega_val)
    print("Rho:", rho_val)
    print("Theta:", theta_val)

    plot_paths, _, _ = GBMwithjump(S0=S0, T=T, M=M, r=R, div=DIV, sigma=SIGMA, simulations=200, mu_extra=ME, lambda_jump=LAMBDAJ, mu_jump=MJ, sigma_jump=SIGMAJ, seed=300)
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Monte Carlo Paths with Jumps/drift (S0={S0}, sigma={SIGMA}, lambda={LAMBDAJ})")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Stock Price")
    time_grid = np.linspace(0, T, M+1)
    for i in range(10):
        plt.plot(time_grid, plot_paths[:, i], lw=1)
    plt.grid(True)
    plt.show()

#different parameters loop (for comparison purpose)
    def prices():
        for S0_val in (45.0, 47.0, 49.0, 52.0, 65.0):
            for vol in (0.1, 0.4):
                for T_val in (1.0, 2.0):
                    price_val = compute_price(OPTION_TYPE, S0_val, STRIKE, T_val, M, R, DIV, vol, 2000, ME, LAMBDAJ, MJ, SIGMAJ)
                    print("S0: %4.1f, Sigma: %4.2f, T: %2.1f --> Option Price: %8.3f" %
                          (S0_val, vol, T_val, price_val))

#timing function

    t0 = time()
    prices()
    t1 = time()
    print("Duration in Seconds: %6.3f" % (t1 - t0))