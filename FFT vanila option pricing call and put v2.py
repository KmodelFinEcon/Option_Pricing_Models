#FFT European option pricing CALL and PUT
#           by. K.tomov

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Global Parameters:

# S0 : Current asset price
#K : Strike price
#T : Time to maturity
#r : Risk-free rate
#sigma : Volatility
#option_type: 'call' or 'put'
#N : Number of FFT discretization points
#delta : Grid spacing in Fourier space
#alpha : Damping factor (if None, alpha is set based on option_type)

# BS objective function

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# Main pricing function via Carr-Madan

def fft_option_price(S0, K, T, r, sigma, option_type='put', N=2**9, delta=0.30, alpha=None):

    if alpha is None:
        alpha = 1.5 if option_type == 'call' else 1.5 

    # Characteristic function of log(ST)
    def char_func(u):
        return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma**2)*T) - 0.5 * sigma**2 * T * u**2)

    lambd = 2 * np.pi / (N * delta)      
    b = N * lambd / 2                    
    v = np.arange(N) * delta             

    simpson_weights = (3 + (-1)**np.arange(N)) / 3.0
    simpson_weights[0] = 1.0 / 3.0

    psi = (np.exp(-r * T) * char_func(v - 1j * (alpha + 1)) /
           (alpha**2 + alpha - v**2 + 1j * (2*alpha + 1) * v))
    
    psi *= np.exp(1j * v * b) * delta * simpson_weights

    fft_values = np.fft.fft(psi).real
    k_grid = -b + lambd * np.arange(N)
    call_prices = np.exp(-alpha * k_grid) * fft_values / np.pi

    logK = np.log(K)
    price_call = np.interp(logK, k_grid, call_prices)

    if option_type.lower() == 'call':
        return price_call
    elif option_type.lower() == 'put':
        price_put = price_call - S0 + K * np.exp(-r * T)
        return price_put
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

if __name__ == '__main__':
    
    # Model test parameters
    r = 0.43
    S0 = 120
    sigma = 0.15
    K = 100
    T = 3
    N = 2**5
    delta = 0.25
    
    analytic_call = black_scholes_call(S0, K, T, r, sigma)
    analytic_put = black_scholes_put(S0, K, T, r, sigma)

    fft_call = fft_option_price(S0, K, T, r, sigma, option_type='call', N=N, delta=delta)
    fft_put  = fft_option_price(S0, K, T, r, sigma, option_type='put', N=N, delta=delta)

    print("Analytic Call Price: ", analytic_call)
    print("FFT Call Price:      ", fft_call)
    print("Analytic Put Price:  ", analytic_put)
    print("FFT Put Price:       ", fft_put)

    lambd = 2 * np.pi / (N * delta)
    b = N * lambd / 2
    v = np.arange(N) * delta
    simpson_weights = (3 + (-1)**np.arange(N)) / 3.0
    simpson_weights[0] = 1.0 / 3.0

    def char_func(u):
        return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma**2) * T) - 0.5 * sigma**2 * T * u**2)
    psi = (np.exp(-r * T) * char_func(v - 1j * (1.5 + 1)) /
           (1.5**2 + 1.5 - v**2 + 1j * (2*1.5 + 1) * v))
    psi *= np.exp(1j * v * b) * delta * simpson_weights
    fft_values = np.fft.fft(psi).real
    k_grid = -b + lambd * np.arange(N)
    call_prices = np.exp(-1.5 * k_grid) * fft_values / np.pi
    strikes = np.exp(k_grid)

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, call_prices, label='FFT Call Prices')
    plt.xlabel('Strike')
    plt.ylabel('Option Price')
    plt.title('FFT Option Price Curve')
    plt.legend()
    plt.show()