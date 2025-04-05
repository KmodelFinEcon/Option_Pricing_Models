
#Tri-Nominal Lattice for Vanilla American Option implementaiton by K.Tomov

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Global parameters

S = 91# starting asset price
K = 95.00 # Strike price 
r = 0.034 #risk free rate
T = 1 #TTM
t = 500 #time steps in lattice simulation 
v = 0.27 #vol
x = 1 #>> call=1, put=-1)

def crr_trinomial_tree(return_tree=False):
    global S, K, r, T, t, v, x
    dt = T / t
    tree = np.empty((2*t-1, t)) # the lattice array
    tree[:] = np.nan
    u = math.exp(v * math.sqrt(2 * dt))
    d = 1 / u

    # Trinomial probabilities based on CRR
    pu = ((math.exp(r * dt/2) - math.exp(-v * math.sqrt(dt/2))) /
          (math.exp(v * math.sqrt(dt/2)) - math.exp(-v * math.sqrt(dt/2)))) ** 2
    pd = ((math.exp(v * math.sqrt(dt/2)) - math.exp(r * dt/2)) /
          (math.exp(v * math.sqrt(dt/2)) - math.exp(-v * math.sqrt(dt/2)))) ** 2
    pm = 1 - (pu + pd)

    mid = t - 1
    lastCol = t - 1

    for row in range(mid - lastCol, mid + lastCol + 1):
        m = row - mid
        if m >= 0:
            asset_price = S * (u ** m)
        else:
            asset_price = S * (d ** (-m))
        tree[row, lastCol] = max(x * (asset_price - K), 0)

    # Backward induction method
    for col in range(lastCol-1, -1, -1):
        for row in range(mid - col, mid + col + 1):
            continuation_value = math.exp(-r * dt) * (
                pu * tree[row-1, col+1] +
                pm * tree[row, col+1] +
                pd * tree[row+1, col+1]
            )
            m = row - mid
            if m >= 0:
                asset_price = S * (u ** m)
            else:
                asset_price = S * (d ** (-m))
            immediate_exercise = max(x * (asset_price - K), 0)
            tree[row, col] = max(immediate_exercise, continuation_value)

    if return_tree:
        return tree, u, d, mid 
    else:
        return tree[mid, 0] 

def black_scholes():
    global S, K, r, T, v, x
    d1 = (math.log(S / K) + (v**2/2 + r) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    return x * S * norm.cdf(d1) - x * K * math.exp(-r * T) * norm.cdf(d2)

crr_price = crr_trinomial_tree()
bs_price  = black_scholes()

print("Option by tri-nominal model ")
print("CRR Call Price: {:3.3f}".format(crr_price))
print("BS Call Price: {:3.3f}".format(bs_price))

lattice, u, d, mid = crr_trinomial_tree(return_tree=True)
times = np.arange(t)
plot_times = []
plot_values = []

for col in range(t):
    for row in range(mid - col, mid + col + 1):
        if not np.isnan(lattice[row, col]):
            plot_times.append(col * (T/t)) 
            plot_values.append(lattice[row, col])

plt.figure(figsize=(9,4))
plt.scatter(plot_times, plot_values, s=2, color='red', alpha=0.6)
plt.title('Trinomial Lattice Option Values in function of timesteps')
plt.xlabel('Time in years')
plt.ylabel('Option value')
plt.grid(True)
plt.show()