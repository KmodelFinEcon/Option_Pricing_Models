
##### AUTOMATIC DIFFERENTIATION PYTORCH #####
#           By. K.T

import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt

K = torch.tensor(100.0, requires_grad=True)
S = torch.tensor(100.0, requires_grad=True)
T = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(0.05, requires_grad=True)
r = torch.tensor(0.01, requires_grad=True)

Z = torch.randn([1000000])
# Brownian Motion
W_T = torch.sqrt(T) * Z
# GBM
prices = S * torch.exp((r - 0.5 * torch.square(sigma)) * T + sigma * W_T)

plt.rcParams["figure.figsize"] = (15, 10)

plt.hist(prices.detach().numpy(), bins=25)
plt.xlabel("Prices")
plt.ylabel("Occurences")
plt.title("Distribution of Underlying Price after 1 Year")

payoffs = torch.max(prices - K, torch.zeros(1000000))
value = torch.mean(payoffs) * torch.exp(-r * T)
print(value)

value.backward()
print(f"Delta: {S.grad}\nVega: {sigma.grad}\nTheta: {T.grad}\nRho: {r.grad}")

# All the same parameters for the price process
K = torch.tensor(100.0, requires_grad=True)
S = torch.tensor(100.0, requires_grad=True)
T = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(0.05, requires_grad=True)
r = torch.tensor(0.01, requires_grad=True)

dt = torch.tensor(1 / 252)
Z = torch.randn([1000000, int(T * 252)])

# Brownian Motion
W_t = torch.cumsum(torch.sqrt(dt) * Z, 1)
# GBM
prices = S * torch.exp((r - 0.5 * torch.square(sigma)) * T + sigma * W_t)

plt.plot(prices[0, :].detach().numpy())
plt.xlabel("Number of Days in Future")
plt.ylabel("Underlying Price")
plt.title("One Possible Price path")
plt.axhline(y=torch.mean(prices[0, :]).detach().numpy(), color="r", linestyle="--")
plt.axhline(y=100, color='g', linestyle="--")