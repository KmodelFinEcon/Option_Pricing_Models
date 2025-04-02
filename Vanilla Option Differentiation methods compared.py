#vanilla option common differentiation methods compared with time to compute

#           by. K.Tomov

import math
import numpy as np
import scipy.stats
from time import time

# Global parameters
stockPrice = 95
strike = 100
rfr = 0.04
dividend = 0.017
Maturity = 1
volatility = 0.214
sigma = volatility  # using volatility as sigma
simulations = 25000
timeSteps = 365
mu = 0.03

# Option type: "call" or "put"
optionType = "call" 

# Closed-form Black-Scholes method

class BlackScholesMethod:
    def compute(self):
        s = stockPrice
        K = strike
        r = rfr
        d = dividend
        T = Maturity
        vol = volatility

        d1 = (math.log(s / K) + (r - d + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        callValue = s * math.exp(-d * T) * scipy.stats.norm.cdf(d1) - K * math.exp(-r * T) * scipy.stats.norm.cdf(d2)
        delta = math.exp(-d * T) * scipy.stats.norm.cdf(d1)
        vega = s * math.exp(-d * T) * scipy.stats.norm.pdf(d1) * math.sqrt(T)
        rho = K * T * math.exp(-r * T) * scipy.stats.norm.cdf(d2)
        
        # Compute put value via put-call parity:
        putValue = callValue - s + K * math.exp(-r * T)
        optionValue = putValue if optionType == "put" else callValue
        
        return {"optionValue": optionValue,
                "callValue": callValue,
                "putValue": putValue,
                "delta": delta,
                "vega": vega,
                "rho": rho}

# Monte Carlo method with adjoint differentiation

class MonteCarloAdjointMethod:
    def compute(self):
        s = stockPrice
        K = strike
        r = rfr
        d = dividend
        T = Maturity
        sim = simulations
        steps = timeSteps
        sigma_local = sigma

        dt = T / steps
        sqrtDt = math.sqrt(dt)
        nudt = (r - d - 0.5 * sigma_local ** 2) * dt
        sigmasdt = sigma_local * sqrtDt
        lnSo = math.log(s)
        discount = math.exp(-r * T)

        sumCallValue = 0.0
        sumDelta = 0.0
        sumR = 0.0
        sumSig = 0.0

        np.random.seed(3000)
        Stockn = np.zeros(steps + 1)
        Stockn[0] = s

        for _ in range(sim):
            lnSt = lnSo
            randVector = np.random.randn(steps)
            # Forward pass: simulate the asset path
            for i in range(1, steps + 1):
                lnSt += nudt + sigmasdt * randVector[i - 1]
                Stockn[i] = math.exp(lnSt)
            endStockValue = Stockn[steps]
            payoff = max(0, endStockValue - K)
            sumCallValue += payoff

            # Only compute sensitivities if the option is in the money
            if endStockValue > K:
                SBar = discount  # sensitivity from final payoff
                rhoBar = -discount * (endStockValue - K) * T
                sigmaBar = 0.0
                # Backward (adjoint) pass
                for n in range(steps - 1, -1, -1):
                    rhoBar += Stockn[n] * dt * SBar
                    sigmaBar += Stockn[n] * sqrtDt * randVector[n] * SBar
                    SBar = (1 + nudt + sigma_local * sqrtDt * randVector[n]) * SBar
                sumDelta += SBar
                sumR += rhoBar
                sumSig += sigmaBar

        callValue = (sumCallValue / sim) * discount
        delta = sumDelta / sim
        rho_val = sumR / sim
        vega = sumSig / sim
        
        # Compute put value using put-call parity
        putValue = callValue - s + K * math.exp(-r * T)
        optionValue = putValue if optionType == "put" else callValue

        return {"optionValue": optionValue,
                "callValue": callValue,
                "putValue": putValue,
                "delta": delta,
                "vega": vega,
                "rho": rho_val}

# Finite Difference method using Monte Carlo

class FiniteDifferenceMethod:
    def MC(self, S, K, r, d, T, sigma, paths, steps):
        lnS = math.log(S)
        np.random.seed(3000)
        dt = T / steps
        nudt = (r - d - 0.5 * sigma ** 2) * dt
        randMatrix = np.random.randn(paths, steps)
        increments = nudt + randMatrix * sigma * math.sqrt(dt)
        lnS_matrix = np.full((paths, 1), lnS)
        lnPaths = np.hstack((lnS_matrix, increments))
        cumPaths = np.cumsum(lnPaths, axis=1)
        expPaths = np.exp(cumPaths)
        finalS = expPaths[:, -1]
        payoffs = np.maximum(finalS - K, 0)
        callValue = np.mean(payoffs) * math.exp(-r * T)
        return callValue

    def compute(self):
        s = stockPrice
        K = strike
        r = rfr
        d = dividend
        T = Maturity
        sigma_local = sigma
        paths = simulations
        steps = timeSteps

        callValue = self.MC(s, K, r, d, T, sigma_local, paths, steps)

        # Delta via finite differences
        tweak = 0.01
        cPlus = self.MC(s + tweak, K, r, d, T, sigma_local, paths, steps)
        cMinus = self.MC(s - tweak, K, r, d, T, sigma_local, paths, steps)
        delta = (cPlus - cMinus) / (2 * tweak)

        # Vega via finite differences
        sigTweak = 0.0001
        cPlusSig = self.MC(s, K, r, d, T, sigma_local + sigTweak, paths, steps)
        cMinusSig = self.MC(s, K, r, d, T, sigma_local - sigTweak, paths, steps)
        vega = (cPlusSig - cMinusSig) / (2 * sigTweak)

        # Rho via finite differences
        rhoTweak = 0.0001
        cPlusRho = self.MC(s, K, r + rhoTweak, d, T, sigma_local, paths, steps)
        cMinusRho = self.MC(s, K, r - rhoTweak, d, T, sigma_local, paths, steps)
        rho_val = (cPlusRho - cMinusRho) / (2 * rhoTweak)
        
        # put value via put-call parity
        putValue = callValue - s + K * math.exp(-r * T)
        optionValue = putValue if optionType == "put" else callValue

        return {"optionValue": optionValue,
                "callValue": callValue,
                "putValue": putValue,
                "delta": delta,
                "vega": vega,
                "rho": rho_val}

#outputs

def main():
    print(f"Option Type: {optionType.upper()}")
    print("Closed-Form Black-Scholes Method:")
    bs = BlackScholesMethod()
    bs_results = bs.compute()
    for key, value in bs_results.items():
        print(f"{key}: {value}")
    print("__________")
    
    print("Monte Carlo Adjoint Method:")
    mc_adj = MonteCarloAdjointMethod()
    t0 = time()
    mc_adj_results = mc_adj.compute()
    for key, value in mc_adj_results.items():
        print(f"{key}: {value}")
    print("Elapsed time: {:.2f} s".format(time() - t0))
    print("____________")
    
    print("Finite Difference Method:")
    fd = FiniteDifferenceMethod()
    t0 = time()
    fd_results = fd.compute()
    for key, value in fd_results.items():
        print(f"{key}: {value}")
    print("Elapsed time: {:.2f} s".format(time() - t0))

if __name__ == '__main__':
    main()