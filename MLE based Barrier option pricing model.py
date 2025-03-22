#barrier option pricing model MLE method#
#           by. k.tomov

import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as sps
import scipy.optimize as spop
import matplotlib.pyplot as plt

# Global parameters
ticker = 'MSFT'
start_date = '2018-03-13'
end_date = '2024-03-13'
strike = 399
barrier_up = 417
barrier_down = 342
rfr = 0.047
maturity = 30  # in days
n_simulations = 1000 #monte carlo simulation of path GBM

# Function to download and process data with error checks
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    if data.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}.")
    prices = data['Close']
    if len(prices) < 2:
        raise ValueError("Insufficient data points to calculate returns.")
    returns = np.array(prices.pct_change().dropna())
    returns = np.sort(returns)
    return prices, returns

# Maximum Likelihood Estimation (MLE) method
def MLEmethod(k, returns):
    k[1] = abs(k[1])
    k[3] = abs(k[3])
    pdf = k[3] / (k[1] * (2 * np.pi) ** 0.5) / (1 + ((returns - k[0]) / k[1]) ** 2) ** 0.5 * np.exp(-0.5 * (k[2] + k[3] * np.arcsinh((returns - k[0]) / k[1])) ** 2)
    return -np.sum(np.log(pdf))

def simulate_prices(prices, su_loc_1, su_scale_1, su_loc_2, su_scale_2, maturity, n_sim):
    if len(prices) == 0:
        raise ValueError("Price data empty. Check data download.")
    initial_price = prices.iloc[-1]
    seed = np.random.random((maturity, n_sim))
    sim_returns = su_loc_1 + su_scale_1 * np.sinh((sps.norm.ppf(seed) - su_loc_2) / su_scale_2)
    sim_prices = initial_price * pd.DataFrame(1 + sim_returns).cumprod()
    return sim_prices

def calculate_option_values(sim_prices, strike, barrier_up, barrier_down, risk_free, maturity):
    max_prices = sim_prices.max()
    min_prices = sim_prices.min()
    price_at_expiry = sim_prices.iloc[-1]

    up_in_payoff = (max_prices >= barrier_up) * np.maximum(0, price_at_expiry - strike)
    up_out_payoff = (max_prices < barrier_up) * np.maximum(0, price_at_expiry - strike)
    down_in_payoff = (min_prices <= barrier_down) * np.maximum(0, strike - price_at_expiry)
    down_out_payoff = (min_prices > barrier_down) * np.maximum(0, strike - price_at_expiry)

    discount_factor = (1 + risk_free) ** (maturity / 252)
    up_in_value = np.average(up_in_payoff) / discount_factor
    up_out_value = np.average(up_out_payoff) / discount_factor
    down_in_value = np.average(down_in_payoff) / discount_factor
    down_out_value = np.average(down_out_payoff) / discount_factor

    return up_in_value, up_out_value, down_in_value, down_out_value

def main():
    try:
        prices, returns = get_data(ticker, start_date, end_date)
        print(f"Successfully downloaded {len(prices)} data points.")
    except Exception as e:
        print(f"Error in data retrieval: {str(e)}")
        return

    # MLE optimization
    try:
        res = spop.minimize(
            MLEmethod, 
            [np.mean(returns), np.std(returns), 0, 1], 
            args=(returns,), 
            method='Nelder-Mead'
        )
        su_loc_1, su_scale_1, su_loc_2, su_scale_2 = res.x[0], abs(res.x[1]), res.x[2], abs(res.x[3])
    except Exception as e:
        print(f"Error in MLE optimization: {str(e)}")
        return

    try:
        sim_prices = simulate_prices(prices, su_loc_1, su_scale_1, su_loc_2, su_scale_2, maturity, n_simulations)
        up_in_value, up_out_value, down_in_value, down_out_value = calculate_option_values(
            sim_prices, strike, barrier_up, barrier_down, rfr, maturity
        )
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        return

    # Print results
    print(f"\nFair values:")
    print(f"- Up-and-in call: ${round(up_in_value, 2)}")
    print(f"- Up-and-out call: ${round(up_out_value, 2)}")
    print(f"- Down-and-in put: ${round(down_in_value, 2)}")
    print(f"- Down-and-out put: ${round(down_out_value, 2)}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sim_prices)
    plt.axhline(barrier_up, color='r', linestyle='--', label='Upper Barrier')
    plt.axhline(barrier_down, color='g', linestyle='--', label='Lower Barrier')
    plt.axhline(strike, color='b', linestyle='--', label='Strike Price')
    plt.title(f"Monte Carlo Simulation Results ({n_simulations} paths)")
    plt.xlabel("Trading Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()