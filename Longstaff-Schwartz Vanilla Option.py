#Longstaff-Schwartz backward vanilla option Call and Put pricing:
#               by Kaloi

import numpy as np
import statsmodels.api as sm

# Global Option Parameters

Spot= 85 #ititial spot price
Strike= 93 #strike price
rfr= 0.04 #risk free rate
Sigma= 0.15 #vol
T= 1 #Time to maturity (in years)
N= 500 #time steps
M= 10000 # simulated paths
k= 3 #Laguerre basis function. NB beteween 1 and 4
Seed= 42 #random seed for loop

OPTION_TYPE = "call"  # either "call" or "put"

def laguerre_basis_functions(k, X):
    #k:  Number of basis functions 1=><4
    #X:  Scaled input array 
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X needs to be array")
    if k < 1 or k > 4:
        raise ValueError("K needs to be between 1 and 4")

    L0 = np.ones_like(X)
    L1 = 1 - X
    L2 = 0.5 * (2 - 4 * X + X**2)
    L3 = (1 / 6) * (6 - 18 * X + 9 * X**2 - X**3)
    L4 = (1 / 24) * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4)
    
    basis = [L0, L1, L2, L3, L4]
    return tuple(basis[:k])

def longstaff_schwartz(N=N, M=M, k=k, seed=Seed):
    
    global Spot, Strike, Sigma, T, rfr, OPTION_TYPE

    np.random.seed(seed)
    dt = T / N
    drift = (rfr - 0.6 * Sigma**2) * dt
    dZ = np.random.normal(0, np.sqrt(dt), size=(M, N))
    diffusion = Sigma * dZ

    # Simulate stock price paths
    stock_paths = np.zeros((M, N + 1))
    stock_paths[:, 0] = Spot
    stock_paths[:, 1:] = Spot * np.cumprod(np.exp(drift + diffusion), axis=1)

    if OPTION_TYPE.lower() == "call":
        option_values = np.maximum(stock_paths - Strike, 0)
    elif OPTION_TYPE.lower() == "put":
        option_values = np.maximum(Strike - stock_paths, 0)
    else:
        raise ValueError("OPTION_TYPE must be either 'call' or 'put'.")

    cash_flows = option_values[:, -1]
    discount_factor = np.exp(-rfr * dt)

    # Main computation of Backward induction using Longstaff-Schwartz
    
    for t in range(N - 1, 0, -1):
        in_the_money = option_values[:, t] > 0
        if not np.any(in_the_money):
            cash_flows *= discount_factor
            continue

        X = stock_paths[in_the_money, t]
        Y = cash_flows[in_the_money] * discount_factor

        basis = np.column_stack(laguerre_basis_functions(k, X / Spot))
        model = sm.OLS(Y, basis).fit()
        continuation_value = model.predict(basis)
        intrinsic_value = option_values[in_the_money, t]
        exercise = intrinsic_value > continuation_value

        cash_flows[in_the_money] = np.where(exercise, intrinsic_value, cash_flows[in_the_money] * discount_factor)
        cash_flows[~in_the_money] *= discount_factor

    mean_option_price = np.mean(cash_flows) * discount_factor
    std_dev = np.std(cash_flows)
    conf_interval = 1.96 * std_dev / np.sqrt(M)

    return {
        "mean": mean_option_price,
        "upper limit": mean_option_price + conf_interval,
        "lower limit": mean_option_price - conf_interval,
        "STDev": std_dev,
    }

#output results

if __name__ == "__main__":
    
    OPTION_TYPE = "call"
    print("Call Option:")
    result_call = longstaff_schwartz()
    print(result_call)

    OPTION_TYPE = "put"
    print("\nPut Option:")
    result_put = longstaff_schwartz()
    print(result_put)
    
    
    