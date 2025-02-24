import torch
from torch.distributions import Normal

std_norm_cdf = Normal(0, 1).cdf
std_norm_pdf = lambda x: torch.exp(Normal(0, 1).log_prob(x))

def bs_price(right, K, S, T, sigma, r):
    d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
    d_2 = d_1 - sigma * torch.sqrt(T)
    
    if right == "C":
        C = std_norm_cdf(d_1) * S - std_norm_cdf(d_2) * K * torch.exp(-r * T)
        return C
        
    elif right == "P":
        P = std_norm_cdf(-d_2) * K * torch.exp(-r * T) - std_norm_cdf(-d_1) * S
        return P
    
right = "C"
K = torch.tensor(120.0, requires_grad=True)
S = torch.tensor(100.0, requires_grad=True)
T = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(0.05, requires_grad=True)
r = torch.tensor(0.01, requires_grad=True)

price = bs_price(right, K, S, T, sigma, r)
print(price)

# Tell PyTorch to compute gradients
price.backward()

print(f"Delta: {S.grad}\nVega: {sigma.grad}\nTheta: {T.grad}\nRho: {r.grad}")

d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
d_2 = d_1 - sigma * torch.sqrt(T)

S = torch.tensor(100.0, requires_grad=True)
price = bs_price(right, K, S, T, sigma, r)

delta = torch.autograd.grad(price, S, create_graph=True)[0]
delta.backward()

print(f"Autograd Gamma: {S.grad}")

# And the direct Black-Scholes calculation
gamma = std_norm_pdf(d_1) / (S * sigma * torch.sqrt(T))
print(f"BS Gamma: {gamma}")