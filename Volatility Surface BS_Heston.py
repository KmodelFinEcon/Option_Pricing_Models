####volatility surface construction####
#       by K.Tomov

import yfinance as yf
import pandas as pd
from scipy.optimize import brentq
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options

####BS Formula####

# Black-Scholes call option pricing formula
def bs_call_price(S, K, T, r, sigma):
    d1 = (log(S/K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

# Implied volatility calculation
def implied_volatility(option_price, S, K, T, r):
    # Use Brent's method to solve for implied volatility
    objective_function = lambda sigma: bs_call_price(S, K, T, r, sigma) - option_price
    return brentq(objective_function, 1e-6, 5)

# Example: Calculate implied volatility for an option

S = 450  # Current stock price (SPY)
K = 455  # Option strike price
T = 30/365  # Time to maturity (30 days)
r = 0.01  # Risk-free interest rate
option_price = 7.5  # Market price of the option

# Calculate implied volatility
iv = implied_volatility(option_price, S, K, T, r)
print(f"Implied Volatility: {iv:.2%}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data for strikes, maturities, and implied volatilities
strikes = np.linspace(440, 460, 10)  # Strike prices
maturities = np.linspace(0.05, 0.5, 10)  # Maturities (in years)

# Simulated implied volatilities (in practice, calculate using real data)
vol_surface = np.random.rand(10, 10) * 0.2 + 0.1

# Create a meshgrid for strikes and maturities
X, Y = np.meshgrid(strikes, maturities)

# Plot the volatility surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, vol_surface, cmap='viridis')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Maturity (Years)')
ax.set_zlabel('Implied Volatility')
ax.set_title(f'Volatility Surface for {ticker}')

plt.show()

#heston Surface

import QuantLib as ql

def calculate_heston_option_price(spot_price, strike_price, maturity, risk_free_rate, dividend_yield, v0, kappa, theta, sigma, rho):
    """
    Calculate the option price using the Heston model.

    Parameters:
    spot_price (float): Spot price of the underlying asset.
    strike_price (float): Strike price of the option.
    maturity (ql.Period): Time to maturity of the option.
    risk_free_rate (float): Risk-free interest rate.
    dividend_yield (float): Dividend yield of the underlying asset.
    v0 (float): Initial variance.
    kappa (float): Mean reversion speed.
    theta (float): Long-term variance.
    sigma (float): Volatility of volatility.
    rho (float): Correlation between the asset price and its variance.

    Returns:
    float: The calculated option price.
    """
    # Construct the Heston process and model
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    heston_process = ql.HestonProcess(
        ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, risk_free_rate, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, dividend_yield, day_count)),
        ql.QuoteHandle(ql.SimpleQuote(spot_price)),
        v0, kappa, theta, sigma, rho
    )

    heston_model = ql.HestonModel(heston_process)
    option_engine = ql.AnalyticHestonEngine(heston_model)

    # Set up the option and calculate the price
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
    exercise = ql.EuropeanExercise(ql.Date().todaysDate() + maturity)

    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(option_engine)

    option_price = option.NPV()
    return option_price

def main():
    # Parameters for the Heston model
    spot_price = 450  # Spot price of the underlying asset
    strike_price = 455  # Strike price
    maturity = ql.Period(30, ql.Days)  # Time to maturity
    risk_free_rate = 0.01  # Risk-free rate
    dividend_yield = 0.02  # Dividend yield
    v0 = 0.1  # Initial variance
    kappa = 2.0  # Mean reversion speed
    theta = 0.1  # Long-term variance
    sigma = 0.2  # Volatility of volatility
    rho = -0.5  # Correlation between the asset price and its variance

    # Calculate the option price
    option_price = calculate_heston_option_price(spot_price, strike_price, maturity, risk_free_rate, dividend_yield, v0, kappa, theta, sigma, rho)
    print(f"Option Price using Heston Model: {option_price}")

if __name__ == "__main__":
    main()