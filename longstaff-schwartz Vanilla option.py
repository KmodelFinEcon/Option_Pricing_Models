#longstaff-schwartz backward vanilla option pricing:
#               by Kaloi

import numpy as np
import statsmodels.api as sm

#Global Parameters
OPTION_PARAMS = {
    "spot": 85, #spot price
    "strike": 93, #strike price
    "r": 0.04, #rfr treasury yield
    "sigma": 0.15, #vol
    "T": 1, #timeframe
}

def basis_function(k, X):
    if not isinstance(X, np.ndarray):
        raise TypeError("np.ndarray")
    if k > 4:
        raise ValueError("The value of k <= 4")

    # Laguerre polynomial basis functions
    basis_funcs = [
        [np.ones_like(X), (1 - X)],
        [np.ones_like(X), (1 - X), 0.5 * (2 - 4 * X + X**2)],
        [
            np.ones_like(X),
            (1 - X),
            0.5 * (2 - 4 * X + X**2),
            (1 / 6) * (6 - 18 * X + 9 * X**2 - X**3),
        ],
        [
            np.ones_like(X),
            (1 - X),
            0.5 * (2 - 4 * X + X**2),
            (1 / 6) * (6 - 18 * X + 9 * X**2 - X**3),
            (1 / 24) * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4),
        ],
    ]
    return tuple(basis_funcs[k - 1])

#objective function

def longstaff_schwartz(spot, strike, sigma, T, r, N=500, M=10000, isCall=True, k=3, seed=42): 
    np.random.seed(seed)
    dt = T / N
    drift = (r - 0.5 * sigma**2) * dt
    dZ = np.random.normal(0, np.sqrt(dt), size=(M, N))
    diffusion = sigma * dZ

    stock_paths = np.zeros((M, N + 1))
    stock_paths[:, 0] = spot
    stock_paths[:, 1:] = spot * np.cumprod(np.exp(drift + diffusion), axis=1)

    option_prices = (
        np.maximum(stock_paths - strike, 0) if isCall else np.maximum(strike - stock_paths, 0)
    )

    cash_flows = option_prices[:, -1]
    discount_factor = np.exp(-r * dt)

    for t in range(N - 1, 0, -1):
        in_the_money = option_prices[:, t] > 0
        if not np.any(in_the_money):
            continue

        X = stock_paths[in_the_money, t]
        Y = cash_flows[in_the_money] * discount_factor

        basis = np.column_stack(basis_function(k, X / spot))
        model = sm.OLS(Y, basis).fit()
        continuationvalue = model.predict(basis)
        intrinsic_value = option_prices[in_the_money, t]
        exercise = intrinsic_value > continuationvalue

        cash_flows[in_the_money] = np.where(
            exercise, intrinsic_value, cash_flows[in_the_money] * discount_factor
        )

    mean_option_price = np.mean(cash_flows) * discount_factor
    std_dev = np.std(cash_flows)
    conf_interval = 1.96 * std_dev / np.sqrt(M)

    return {
        "mean_option_price": mean_option_price,
        "upper_limit": mean_option_price + conf_interval,
        "lower_limit": mean_option_price - conf_interval,
        "std dev": std_dev,
    }

if __name__ == "__main__":
    result = longstaff_schwartz(**OPTION_PARAMS)
    print(result)
    
    