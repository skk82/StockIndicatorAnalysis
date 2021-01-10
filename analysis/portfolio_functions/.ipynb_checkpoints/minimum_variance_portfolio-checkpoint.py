import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_minimum_variance_frontier(self, column='Close', *, industry='ALL', iters=5000, points=300,
                                       figsize=(12, 8), seed=42):
        np.random.seed(seed)
        if industry == 'ALL':
            df = self.df.pivot_table(values=column, index='Date', columns='ticker')
        else:
            df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
        log_ret = np.log(df / df.shift(1))

        weights, rets, vols, sharpe = self._sim_portfolios(log_ret, iters)

        max_ret = rets[sharpe.argmax()]
        max_vol = vols[sharpe.argmax()]

        # max_sharpe_port = self._calc_ret_vol_sharpe(log_ret, self.max_sharpe_portfolio(log_ret=log_ret))

        frontier_returns = np.linspace(0, np.max(rets), points)
        frontier_vols = np.array([])

        for ret in frontier_returns:
            ret_weights = self.weights_for_return(ret, log_ret=log_ret)
            frontier_vols = np.append(frontier_vols, self._calc_ret_vol_sharpe(log_ret, ret_weights)[1])

        plt.figure(figsize=figsize)
        plt.scatter(vols, rets, c=sharpe, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.scatter(max_vol, max_ret, c='red', s=50)

        plt.plot(frontier_vols, frontier_returns, 'o:', linewidth=2)
        plt.show()

    def max_sharpe_portfolio(self, *, column='Close', industry='ALL', log_ret=None):
        if not log_ret:
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            log_ret = np.log(df / df.shift(1))
        print()
        neg_sharpe = lambda weights: self._calc_ret_vol_sharpe(log_ret, weights)[2]
        guess = np.ones(df.shape[1]) / df.shape[1]
        bounds = ((0, 1),) * df.shape[1]
        constraints = ({'type': 'eq', 'fun': lambda weights: 1 - np.sum(weights)},)
        results = minimize(neg_sharpe, guess, method='SLSQP', options={'disp': False}, bounds=bounds,
                           constraints=constraints)
        return results.x

    def weights_for_return(self, ret_val, *, column='Close', industry='ALL', log_ret=None):
        if not log_ret:
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            log_ret = np.log(df / df.shift(1))
        min_vol = lambda weights: self._calc_ret_vol_sharpe(log_ret)[1]
        guess = np.ones(df.shape[1]) / df.shape[1]
        bounds = ((0, 1),) * df.shape[1]
        constraints = ({'type': 'eq', 'fun': lambda weights: 1 - np.sum(weights)},
                       {'type': 'eq', 'fun': lambda weights: self._calc_ret_vol_sharpe(log_ret, weights)[0]-ret_val})
        results = minimize(min_vol, guess, method='SLSQP', options={'disp': False}, bounds=bounds,
                           constraints=constraints)
        return results.x

    def _sim_portfolios(self, log_ret, iters=5000):
        weights = np.zeros((iters, log_ret.shape[1]))
        rets = np.zeros(iters)
        vols = np.zeros(iters)
        sharpe = np.zeros(iters)

        for i in range(iters):
            whole_weights = np.random.random(log_ret.shape[1])
            weights[i, :] = whole_weights / np.sum(whole_weights)

            rets[i], vols[i], sharpe[i] = self._calc_ret_vol_sharpe(log_ret, weights[i, :])

        return weights, rets, vols, sharpe

    @staticmethod
    def _calc_ret_vol_sharpe(log_ret, weights):
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sharpe = ret / vol
        return ret, vol, sharpe


