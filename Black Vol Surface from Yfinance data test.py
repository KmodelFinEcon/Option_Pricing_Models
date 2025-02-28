
###UN-Callibrated Black VOl surface from y-finance data ###

import QuantLib as ql
import math
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import yfinance as yf
from scipy.interpolate import UnivariateSpline

# Defining market data and settings:
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
calculation_date = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = calculation_date

spot = 90.37
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
risk_free_rate = 0.01
dividend_rate = 0.0
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count))

# Retrieve option data using yfinance:
ticker = 'CL'
stock = yf.Ticker(ticker)

expiration_dates_raw = stock.options
expiration_dates = [
    ql.Date(int(date.split('-')[2]), int(date.split('-')[1]), int(date.split('-')[0]))
    for date in expiration_dates_raw
]
# Preserve the original list for further processing:
ql_expiration_dates = expiration_dates.copy()

option_data = stock.option_chain(expiration_dates_raw[0])
calls = option_data.calls
puts = option_data.puts

# Extract strikes from the calls data.
strikes = list(calls['strike'])
strikes_list = strikes.copy()

# Generate a volatility matrix
realized_volatility = np.random.uniform(0.2, 0.4, (len(ql_expiration_dates), len(strikes_list)))
vol_data_matrix = realized_volatility.tolist()

exp_strs = [
    f"ql.Date({exp.dayOfMonth()}, {exp.month()}, {exp.year()})" for exp in ql_expiration_dates
]
formatted_expiration_dates = "expiration_dates = [\n    " + ",\n    ".join(exp_strs) + "\n]"
formatted_strikes = "strikes = " + str(strikes_list)
formatted_data = "data = [\n    " + ",\n    ".join(str(row) for row in vol_data_matrix) + "\n]"

print(formatted_expiration_dates)
print()
print(formatted_strikes)
print()
print(formatted_data)

# --- Smoothing the volatility data - Cubic Spline interpolation
smoothed_vols_list = []
for i in range(len(ql_expiration_dates)):
    strikes_arr = np.array(strikes_list)
    vols_arr = np.array(vol_data_matrix[i])
    spline = UnivariateSpline(strikes_arr, vols_arr, s=0.03)
    smoothed_vol = spline(strikes_arr)
    smoothed_vols_list.append(smoothed_vol.tolist())


vol_surface_matrix = ql.Matrix(len(strikes_list), len(ql_expiration_dates))
for i in range(len(strikes_list)):      # rows correspond to strikes
    for j in range(len(ql_expiration_dates)):  # columns correspond to expiries
        vol_surface_matrix[i][j] = smoothed_vols_list[j][i]

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar, 
    ql_expiration_dates, strikes_list, 
    vol_surface_matrix, day_count)

black_var_surface.enableExtrapolation()

example_strike = 94.0
example_expiry = 1  # years
print(f"Black Volatility for Strike={example_strike}, Expiry={example_expiry}: {black_var_surface.blackVol(example_expiry, example_strike)}")

strikes_grid = np.arange(strikes_list[0], strikes_list[-1], 10)
expiry_for_plot = 1.0  # years
implied_vols_plot = [black_var_surface.blackVol(expiry_for_plot, s) for s in strikes_grid]
index = min(10, len(vol_data_matrix) - 1)
actual_data = vol_data_matrix[index]

fig, ax = plt.subplots()
ax.plot(strikes_grid, implied_vols_plot, label="Black Surface")
ax.plot(strikes_list, actual_data, "o", label="Actual")
ax.set_xlabel("Strikes", size=12)
ax.set_ylabel("Volatility", size=12)
ax.legend(loc="upper right")
plt.show()

max_time = black_var_surface.maxTime()
plot_years = np.linspace(0, max_time, num=20)
plot_strikes = np.arange(min(strikes_list), max(strikes_list) + 1, 1)
X, Y = np.meshgrid(plot_strikes, plot_years)
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = np.array([black_var_surface.blackVol(y, float(x)) for x, y in zip(X_flat, Y_flat)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(X_flat, Y_flat, Z_flat, cmap=cm.magma, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Strikes")
ax.set_ylabel("Time to Maturity (years)")
ax.set_zlabel("Black Volatility")
plt.show()