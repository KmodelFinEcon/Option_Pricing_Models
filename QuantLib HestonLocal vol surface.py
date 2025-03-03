#####QuantLib Mixed Heston-Dupire vol surface calibration vs finite differeence call option #####
#               by kt

import QuantLib as ql
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

#global parameters: 

spot = 120
rate = 0.04
today = ql.Date(1, 7, 2023)
calendar = ql.NullCalendar()
day_count = ql.Actual365Fixed()

# Set up the flat risk-free curves
riskFreeCurve = ql.FlatForward(today, rate, ql.Actual365Fixed())
flat_ts = ql.YieldTermStructureHandle(riskFreeCurve)
dividend_ts = ql.YieldTermStructureHandle(riskFreeCurve)

#Stochastic path generation
num_paths = 2500
timestep = 25
length = 1.0
time_grid = ql.TimeGrid(length, timestep)

#Objective function

def plot_vol_surface(vol_surface, plot_years=np.arange(0.1, 3, 0.1), 
                     plot_strikes=np.arange(70, 130, 1), funct='blackVol'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(plot_strikes, plot_years)

    for surface in (vol_surface if isinstance(vol_surface, list) else [vol_surface]):
        method_to_call = getattr(surface, funct)
        Z = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = method_to_call(float(plot_years[i]), float(plot_strikes[j]))
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def generate_multi_paths_df(sequence, num_paths, time_grid=None):
    spot_paths = []
    vol_paths = []

    for i in range(num_paths):
        sample_path = sequence.next()
        spots, vols = sample_path.value()
        spot_paths.append(list(spots))
        vol_paths.append(list(vols))

    if time_grid is not None:
        columns = [time_grid[i] for i in range(len(spot_paths[0]))]
    else:
        columns = list(range(len(spot_paths[0])))
        
    df_spot = pd.DataFrame(spot_paths, columns=columns)
    df_vol  = pd.DataFrame(vol_paths, columns=columns)
    return df_spot, df_vol

# Vol surface from Heston parameters p.1
def create_vol_surface_mesh_from_heston_params(today, calendar, spot, v0, kappa, theta, rho, sigma, 
                                               rates_curve_handle, dividend_curve_handle,
                                               strikes=np.linspace(40, 200, 161), 
                                               tenors=np.linspace(0.1, 3, 60)):
    quote = ql.QuoteHandle(ql.SimpleQuote(spot))
    heston_process = ql.HestonProcess(rates_curve_handle, dividend_curve_handle, quote, v0, kappa, theta, sigma, rho)
    heston_model = ql.HestonModel(heston_process)
    heston_handle = ql.HestonModelHandle(heston_model)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)

    data = [] #tenor computation
    for strike in strikes:
        data.append([heston_vol_surface.blackVol(tenor, strike) for tenor in tenors])
    expiration_dates = [calendar.advance(today, ql.Period(int(365*t), ql.Days)) for t in tenors]
    implied_vols = ql.Matrix(data)
    feller = 2 * kappa * theta - sigma ** 2

    return expiration_dates, strikes, implied_vols, feller

#constructing vol surface p.2

dates, strikes, vols, feller = create_vol_surface_mesh_from_heston_params(
    today, calendar, spot, 0.0225, 1.0, 0.0625, -0.25, 0.3, flat_ts, dividend_ts)

local_vol_surface = ql.BlackVarianceSurface(today, calendar, dates, strikes, vols, day_count)
plot_vol_surface(local_vol_surface)

spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
local_vol_surface.setInterpolation("bicubic")
local_vol_handle = ql.BlackVolTermStructureHandle(local_vol_surface)
local_vol = ql.LocalVolSurface(local_vol_handle, flat_ts, dividend_ts, spot_quote)
local_vol.enableExtrapolation()

plot_vol_surface(local_vol, funct='localVol') #DUPIRE local Vol

#Calibrating Heston:

spot_heston = 107# from different spot value for comparison
spot_quote_heston = ql.QuoteHandle(ql.SimpleQuote(spot_heston))
v0 = 0.017; kappa = 2.0; theta = 0.045; rho = -0.2; sigma = 0.25
feller = 2 * kappa * theta - sigma ** 2

heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_quote_heston, v0, kappa, theta, sigma, rho)
heston_model = ql.HestonModel(heston_process)
heston_handle = ql.HestonModelHandle(heston_model)
heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)

# Compare local and Heston surfaces:
plot_vol_surface([local_vol_surface, heston_vol_surface])

#calibrating via monte-carlo to get leverage effect L(s,t) 

end_date = ql.Date(1, 7, 2024)
generator_factory = ql.MTBrownianGeneratorFactory(43)
calibration_paths_vars = [2**15, 2**17, 2**19, 2**20]
time_steps_per_year, n_bins = 365, 180

for calibration_paths in calibration_paths_vars:
    print("Paths: {}".format(calibration_paths))
    stoch_local_mc_model = ql.HestonSLVMCModel(local_vol, heston_model, generator_factory, 
                                               end_date, time_steps_per_year, n_bins, calibration_paths)
    a = time.time()
    leverage_function = stoch_local_mc_model.leverageFunction()
    b = time.time()
    print("Calibration took {0:2.1f} seconds".format(b - a))
    plot_vol_surface(leverage_function, funct='localVol', plot_years=np.arange(0.1, 0.98, 0.1))
    plt.pause(0.05)


stoch_local_process = ql.HestonSLVProcess(heston_process, leverage_function)
dimension = stoch_local_process.factors()

rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(dimension * timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianMultiPathGenerator(stoch_local_process, list(time_grid), rng, False)

df_spot, df_vol = generate_multi_paths_df(seq, num_paths, time_grid=time_grid)

# sample simulated paths
fig = plt.figure(figsize=(18, 15))


plt.subplot(2, 2, 1)
plt.plot(df_spot.iloc[0:10].transpose())
plt.title("Spot Paths")

plt.subplot(2, 2, 2)
plt.hist(df_spot.iloc[:, -1], bins=50)
plt.title("Spot Hist at maturity")

plt.subplot(2, 2, 4)
plt.hist(df_vol.iloc[:, -1], bins=50)
plt.title("Vol Hist at maturity")

plt.subplot(2, 2, 3)
plt.plot(df_vol.iloc[0:10].transpose())
plt.title("Vol Paths")


plt.tight_layout()
plt.show()

# (Extra)Pricing a European Call Option using Monte Carlo and Finite Differences

mc_payoffs = (df_spot.iloc[:, -1] - 100).clip(lower=0)
mc_price = np.mean(mc_payoffs)
print("MC call price:", mc_price)

strike = 120
maturity_date = ql.Date(1, 7, 2024)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
exercise = ql.EuropeanExercise(maturity_date)
option = ql.VanillaOption(payoff, exercise)

slv_engine = ql.FdHestonVanillaEngine(heston_model, 200, 200, 100, 0, 
                                      ql.FdmSchemeDesc.Hundsdorfer(), leverage_function)
option.setPricingEngine(slv_engine)
fd_price = option.NPV()
print("FD call price:", fd_price)