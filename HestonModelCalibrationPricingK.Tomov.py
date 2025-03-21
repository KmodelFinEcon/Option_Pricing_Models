
#heston model - Calibration + pricing by K.Tomov

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt

from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

import plotly.graph_objects as go
from plotly.graph_objs import Surface
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

#defining function

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

    return exp1*term2*exp2

#defining integrand

def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi=umax/N #dphi is width

    for i in range(1,N):
        # rectangular integration
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    real_integral, err = np.real( quad(integrand, 0, 100, args=args) )

    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

# Testing parameters

S0 = 100. # initial asset price
K = 100. # strike
v0 = 0.1 # initial variance
r = 0.03 # risk free rate
kappa = 1.5768 # rate of mean reversion of variance process
theta = 0.0398 # long-term mean variance
sigma = 0.3 # volatility of volatility
lambd = 0.575 # risk premium of variance
rho = -0.5711 # correlation between variance and stock process
tau = 1. # time to maturity

heston_price( S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r )

#term structure expansion:

yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yeilds = np.array([0.15,0.27,0.50,0.93,1.52,2.13,2.32,2.34,2.37,2.32,2.65,2.52]).astype(float)/100

#NSS model calibrate
curve_fit, status = calibrate_nss_ols(yield_maturities,yeilds)

curve_fit

#loading data: Yfinance or beautiful soup here >>>>>

resp = yf.getstockoptions('GSPC.INDX')

resp

market_prices = {}

S0 = resp['lastTradePrice']

for i in resp['data']:
    market_prices[i['expirationDate']] = {}
    market_prices[i['expirationDate']]['strike'] = [name['strike'] for name in i['options']['CALL']]# if name['volume'] is not None]
    market_prices[i['expirationDate']]['price'] = [(name['bid']+name['ask'])/2 for name in i['options']['CALL']]# if name['volume'] is not None]
    
all_strikes = [v['strike'] for i,v in market_prices.items()]
common_strikes = set.intersection(*map(set,all_strikes))
print('Number of common strikes:', len(common_strikes))
common_strikes = sorted(common_strikes)

prices = []
maturities = []

for date, v in market_prices.items():
    maturities.append((dt.strptime(date, '%Y-%m-%d') - dt.today()).days/365.25)
    price = [v['price'][i] for i,x in enumerate(v['strike']) if x in common_strikes]
    prices.append(price)

price_arr = np.array(prices, dtype=object)
np.shape(price_arr)

#VOLATILITY SURFACE 

volSurface = pd.DataFrame(price_arr, index = maturities, columns = common_strikes)
volSurface = volSurface.iloc[(volSurface.index > 0.04) & (volSurface.index < 1), (volSurface.columns > 3000) & (volSurface.columns < 5000)]
volSurface

# Convert our vol surface to dataframe for each option price with parameters
volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
volSurfaceLong.columns = ['maturity', 'strike', 'price']

# Calculate the risk free rate for each maturity using the fitted yield curve
volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)

######calibration function######

# heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r) Parameters are v0, kappa, theta, sigma, rho, lambd


# Define variables to be used in optimization
S0 = resp['lastTradePrice']
r = volSurfaceLong['rate'].to_numpy('float')
K = volSurfaceLong['strike'].to_numpy('float')
tau = volSurfaceLong['maturity'].to_numpy('float')
P = volSurfaceLong['price'].to_numpy('float')

params = {"v0": {"x0": 0.1, "lbub": [1e-3,0.1]},
          "kappa": {"x0": 3, "lbub": [1e-3,5]},
          "theta": {"x0": 0.05, "lbub": [1e-3,0.1]},
          "sigma": {"x0": 0.3, "lbub": [1e-2,1]},
          "rho": {"x0": -0.8, "lbub": [-1,0]},
          "lambd": {"x0": 0.03, "lbub": [-1,1]},
          }

x0 = [param["x0"] for key, param in params.items()]
bnds = [param["lbub"] for key, param in params.items()]

def SqErr(x):
    v0, kappa, theta, sigma, rho, lambd = [param for param in x]

    # Attempted to use scipy integrate quad module as constrained to single floats not arrays
    # err = np.sum([ (P_i-heston_price(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i))**2 /len(P) \
    #               for P_i, K_i, tau_i, r_i in zip(marketPrices, K, tau, r)])

    # Decided to use rectangular integration function in the end
    err = np.sum( (P-heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r))**2 /len(P) )

    # Zero penalty term - no good guesses for parameters
    pen = 0 #np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] )

    return err + pen

result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=bnds)

result

v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
v0, kappa, theta, sigma, rho, lambd

####COMPUTATION######

heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)

volSurfaceLong['heston_price'] = heston_prices

volSurfaceLong

fig = go.Figure(data=[go.Mesh3d(x=volSurfaceLong.maturity, y=volSurfaceLong.strike, z=volSurfaceLong.price, color='mediumblue', opacity=0.55)])

fig.add_scatter3d(x=volSurfaceLong.maturity, y=volSurfaceLong.strike, z=volSurfaceLong.heston_price, mode='markers')

fig.update_layout(
    title_text='Market Prices vs Calibrated Heston model',
    scene = dict(xaxis_title='TIME (Years)',
                    yaxis_title='STRIKES (Pts)',
                    zaxis_title='INDEX OPTION PRICE (Pts)'),
    height=800,
    width=800
)

fig.show(renderer="colab")