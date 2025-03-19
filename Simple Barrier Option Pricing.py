# Simple Barrier option pricing model #
#       by K.Tomov

import numpy as np

# Global parameters

S0 = 100 # Initial stock price
K = 100  # Strike price
H_up = 125 # Barrier level for up barrier options
H_down = 75 # Barrier level for down barrier options
T = 1.0 # Time to maturity (years)
r = 0.01 # Risk-free interest rate
sigma = 0.2 # Volatility
N = 100 # Number of time steps
M = 10000 # Number of simulation paths
use_antithetic = False  # Flag for using antithetic variates


#objective function

def barrier_option(option_type="call", barrier_type="up-and-in", **params):

    S0 = params.get("S0", 100)
    K = params.get("K", 100)
    H = params.get("H", 125)
    T = params.get("T", 1.0)
    r = params.get("r", 0.01)
    sigma = params.get("sigma", 0.2)
    N = params.get("N", 100)
    M = params.get("M", 10000)
    use_antithetic = params.get("use_antithetic", False)
    
    # Barrier constraints
    if barrier_type.startswith("up") and S0 >= H:
        raise ValueError("For an 'up' barrier, the barrier level H must be above the initial price S0.")
    if barrier_type.startswith("down") and S0 <= H:
        raise ValueError("For a 'down' barrier, the barrier level H must be below the initial price S0.")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'.")
    
    dt = T / N
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Monte-Carlo simulation
    if use_antithetic:
        if M % 2 != 0:
            M -= 1  # Ensure even number of paths.
        half_M = M // 2
        Z = np.random.standard_normal((half_M, N))
        Z_full = np.concatenate([Z, -Z], axis=0)
    else:
        Z_full = np.random.standard_normal((M, N))
    
    logS = np.cumsum(drift + diffusion * Z_full, axis=1)    # Simulate asset paths: S_t = S0 * exp(cumsum(drift + diffusion * Z))
    S = S0 * np.exp(logS)
    S = np.hstack((S0 * np.ones((M, 1)), S))# Prepend S0 to each path to include the initial value
    
    # Barrier upper and lower hit 
    if "up" in barrier_type:
        barrier_triggered = np.any(S >= H, axis=1)
    elif "down" in barrier_type:
        barrier_triggered = np.any(S <= H, axis=1)
    
    # Payoff based on option type.
    if option_type == "call":
        payoff = np.maximum(S[:, -1] - K, 0)
    else:
        payoff = np.maximum(K - S[:, -1], 0)
    
    # knock-in vs knock-out
    if "in" in barrier_type:
        payoff = payoff * barrier_triggered  # Only paths where barrier was hit
    else:
        payoff = payoff * (~barrier_triggered)  # Only paths where barrier was not hit
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

if __name__ == "__main__":
    price_up_in_call = barrier_option(
        option_type="call", 
        barrier_type="up-and-in", 
        S0=S0, K=K, H=H_up, T=T, r=r, sigma=sigma, N=N, M=M, use_antithetic=use_antithetic
    )
    print("Up-and-in Call:", price_up_in_call)

    # up-and-in put pricing using the global parameters
    price_up_in_put = barrier_option(
        option_type="put", 
        barrier_type="up-and-in", 
        S0=S0, K=K, H=H_up, T=T, r=r, sigma=sigma, N=N, M=M, use_antithetic=use_antithetic
    )
    print("Up-and-in Put:", price_up_in_put)

    # down-and-in call pricing using the global parameters (using H_down)
    price_down_in_call = barrier_option(
        option_type="call", 
        barrier_type="down-and-in", 
        S0=S0, K=K, H=H_down, T=T, r=r, sigma=sigma, N=N, M=M, use_antithetic=use_antithetic
    )
    print("Down-and-in Call:", price_down_in_call)

    #  down-and-in put using the global parameters (using H_down)
    price_down_in_put = barrier_option(
        option_type="put", 
        barrier_type="down-and-in", 
        S0=S0, K=K, H=H_down, T=T, r=r, sigma=sigma, N=N, M=M, use_antithetic=use_antithetic
    )
    print("Down-and-in Put:", price_down_in_put)

    # antithetic variates
    if use_antithetic:
        price_antithetic = barrier_option(
            option_type="call", 
            barrier_type="up-and-in", 
            S0=S0, K=K, H=H_up, T=T, r=r, sigma=sigma, N=N, M=M, use_antithetic=use_antithetic
        )
        print("Up-and-in Call with antithetic variates:", price_antithetic)
