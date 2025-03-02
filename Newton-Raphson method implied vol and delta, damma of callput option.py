###Implied vol using Newton-Raphson method from BS and delta,gamma of call/put option ###
#           by K.Tomov


import numpy as np
import scipy.stats as st

# Parameters
V_market = 2    # market call option price
K = 125         # strike
tau = 1         # time-to-maturity
r = 0.04        # interest rate
S_0 = 109       # today's stock price
sigma_init = 0.084  # Initial implied volatility
CP = "c"        # C is call and P is put

def implied_volatility(CP, S_0, K, sigma, tau, r, max_iter=1000, tol=1e-10): #Calculate the implied volatility using the Newton-Raphson method.
    """
    Calculate the implied volatility using the Newton-Raphson method.
    
    :param CP: Option type ('c' for call, 'p' for put)
    :param S_0: Current stock price
    :param K: Strike price
    :param sigma: Initial guess for volatility
    :param tau: Time to maturity
    :param r: Risk-free interest rate
    :param max_iter: Maximum number of iterations
    :param tol: Tolerance for convergence
    :return: Implied volatility
    """
    error = 1e10  #errors term
    n = 0
    
    while error > tol and n < max_iter:
        opt_price = bs_option_price(CP, S_0, K, sigma, tau, r)
        vega_val = vega(S_0, K, sigma, tau, r)
        
        if vega_val == 0:
            raise ValueError("Vega is zero. Cannot continue.")
        
        g = opt_price - V_market
        sigma_new = sigma - g / vega_val
        
        error = abs(g)
        sigma = sigma_new
        
        print(f'Iteration {n} with error = {error}')
        
        n += 1
    
    if n == max_iter:
        print("Warning: Maximum iterations reached. Result may not be accurate.")
    
    return sigma

def vega(S_0, K, sigma, tau, r):
    d2 = (np.log(S_0 / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    value = K * np.exp(-r * tau) * st.norm.pdf(d2) * np.sqrt(tau)
    return value

def gamma(S_0, K, sigma, tau, r):

    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    value = st.norm.pdf(d1) / (S_0 * sigma * np.sqrt(tau))  # Gamma formula
    return value

def option_delta(CP, S_0, K, sigma, tau, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    
    if CP.lower() == "c":
        delta = st.norm.cdf(d1)  # Delta for call option
    elif CP.lower() == "p":
        delta = st.norm.cdf(d1) - 1  # Delta for put option
    else:
        raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")
    
    return delta

def bs_option_price(CP, S_0, K, sigma, tau, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if CP.lower() == "c":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP.lower() == "p":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    else:
        raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")
    
    return value

# Implied volatility
sigma_imp = implied_volatility(CP, S_0, K, sigma_init, tau, r)

message = '''Implied volatility for Call Price= {}, strike K={}, 
      maturity T= {}, interest rate r= {} and initial stock S_0={} 
      equals to sigma_imp = {:.7f}'''.format(V_market, K, tau, r, S_0, sigma_imp)
print(message)

# CHECKING RESULTS
val = bs_option_price(CP, S_0, K, sigma_imp, tau, r)
print('Option Price for implied volatility of {0} is equal to {1}'.format(sigma_imp, val))

val = bs_option_price(CP, S_0, K, sigma_init, tau, r)
print('Option Price for starting volatility of {0} is equal to {1}'.format(sigma_imp, val))

# DELTA
delta = option_delta(CP, S_0, K, sigma_imp, tau, r)
print(f'The delta with implied vol of the {CP.upper()} option is: {delta:.4f}')

delta = option_delta(CP, S_0, K, sigma_init, tau, r)
print(f'The starting delta of the {CP.upper()} option is: {delta:.4f}')

# GAMMA
gamma_value = gamma(S_0, K, sigma_imp, tau, r)
print(" implied vol Gamma:", gamma_value)

gamma_value = gamma(S_0, K, sigma_init, tau, r)
print("starting Gamma:", gamma_value)