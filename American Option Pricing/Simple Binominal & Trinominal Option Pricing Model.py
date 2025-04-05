### Simple Binominal & Trinominal Option pricing model ###
#            by K.Tomov 

import numpy as np

# Initialise parameters
S0 = 100      # initial stock price
K = 100       # strike price
T = 1         # time to maturity in years
r = 0.06      # annual risk-free rate
N = 3         # number of time steps
u = 1.1       # up-factor in binomial models
d = 1/u       # ensure recombining tree
opttype = 'C' # Option Type 'C' or 'P'

# Binomial Tree Model (Slow Version)
def binomial_tree_slow(K, T, S0, r, N, u, d, opttype='C'):
    # Precompute constants
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity - Time step N
    S = np.zeros(N + 1)
    S[0] = S0 * d**N
    for j in range(1, N + 1):
        S[j] = S[j - 1] * u / d

    # Initialize option values at maturity
    C = np.zeros(N + 1)
    for j in range(0, N + 1):
        if opttype == 'C':
            C[j] = max(0, S[j] - K)  # Call option
        else:
            C[j] = max(0, K - S[j])  # Put option

    # Step backwards through the tree
    for i in np.arange(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (q * C[j + 1] + (1 - q) * C[j])

    return C[0]

# Binomial Tree Model (Fast Version)
def binomial_tree_fast(K, T, S0, r, N, u, d, opttype='C'):
    # Precompute constants
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity - Time step N
    C = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

    # Initialize option values at maturity
    if opttype == 'C':
        C = np.maximum(C - K, np.zeros(N + 1))  # Call option
    else:
        C = np.maximum(K - C, np.zeros(N + 1))  # Put option

    # Step backwards through the tree
    for i in np.arange(N, 0, -1):
        C = disc * (q * C[1:i + 1] + (1 - q) * C[0:i])

    return C[0]

# Trinomial Tree Model
def trinomial_tree(K, T, S0, r, N, u, d, opttype='C'):
    # Precompute constants
    dt = T / N
    pu = 0.3  # Probability of up movement
    pd = 0.3  # Probability of down movement
    pm = 1 - pu - pd  # Probability of staying the same
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity - Time step N
    S = np.zeros((2 * N + 1, N + 1))
    S[N, 0] = S0
    for j in range(1, N + 1):
        for i in range(N - j, N + j + 1):
            if i == N - j:
                S[i, j] = S[i + 1, j - 1] * d
            elif i == N + j:
                S[i, j] = S[i - 1, j - 1] * u
            else:
                S[i, j] = S[i, j - 1]

    # Initialize option values at maturity
    C = np.zeros((2 * N + 1, N + 1))
    for i in range(0, 2 * N + 1):
        if opttype == 'C':
            C[i, N] = max(0, S[i, N] - K)  # Call option
        else:
            C[i, N] = max(0, K - S[i, N])  # Put option

    # Step backwards through the tree
    for j in np.arange(N - 1, -1, -1):
        for i in range(N - j, N + j + 1):
            C[i, j] = disc * (pu * C[i + 1, j + 1] + pm * C[i, j + 1] + pd * C[i - 1, j + 1])

    return C[N, 0]

# Test the models
for N in [3, 50, 100, 1000, 5000]:
    print(f"Number of steps: {N}")
    print(f"Binomial Slow: {binomial_tree_slow(K, T, S0, r, N, u, d, opttype)}")
    print(f"Binomial Fast: {binomial_tree_fast(K, T, S0, r, N, u, d, opttype)}")
    if N <= 100:  # Trinomial is computationally expensive for large N
        print(f"Trinomial: {trinomial_tree(K, T, S0, r, N, u, d, opttype)}")
    print("-" * 50)
    
print(f"if {'binomial_tree_slow'} is {'binomial_tree_fast'} then we are good")
