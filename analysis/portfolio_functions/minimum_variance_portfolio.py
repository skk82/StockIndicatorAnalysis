from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from tqdm import tqdm


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.__log_ret = None

    def plot_minimum_variance_frontier(self, column='Close', *, industry='ALL', iters=5000, points=300,
                                       figsize=(12, 8), seed=42):
        np.random.seed(seed)
        if industry == 'ALL':
            df = self.df.pivot_table(values=column, index='Date', columns='ticker')
        else:
            df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')

        # log_ret = np.log(df / df.shift(1))
        log_ret = df.pct_change()
        self.__log_ret = log_ret

        weights, rets, vols, sharpe = self._sim_portfolios(log_ret, iters)

        ret_vol = rets[vols.argmin()]
        vol_vol = vols.min()

        min_var = weights[vols.argmin()]
        max_sharpe = weights[sharpe.argmax()]

        ret_sharpe, vol_sharpe, _ = self._calc_ret_vol_sharpe(log_ret, self.max_sharpe_portfolio().x)

        frontier_returns = np.linspace(rets.min(), rets.max(), points)

        with Pool(processes=cpu_count() - 1) as pool:
            ret_weights = list(tqdm(pool.imap_unordered(self.weights_for_return, frontier_returns),
                                    desc='Creating Frontier', total=points))

        frontier_vols = [self._calc_ret_vol_sharpe(log_ret, w)[1] for w in ret_weights]

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=figsize)
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(vols, rets, c=sharpe, cmap='YlGnBu', s=50, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')

        plt.scatter(vol_sharpe, ret_sharpe, marker='*', c='red', s=250, label='Maximum Sharpe Ratio')
        plt.scatter(vol_vol, ret_vol, marker='*', c='orange', s=250, label='Minimum Variance Portfolio')

        plt.plot(frontier_vols, frontier_returns, ':', color='orange', linewidth=2)

        plt.legend(labelspacing=0.5, loc='best')
        plt.show()

    def max_sharpe_portfolio(self, *, column='Close', industry='ALL', log_ret=None):
        if not isinstance(log_ret, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            # log_ret = np.log(df / df.shift(1))
            log_ret = df.pct_change()

        def neg_sharpe(weights):
            return -self._calc_ret_vol_sharpe(log_ret, weights)[2]  # Maximize Sharpe by minimizing negative Sharpe

        guess = np.ones(log_ret.shape[1]) / log_ret.shape[1]  # Initial guess of equal weights
        bounds = tuple((0, 1) for _ in range(log_ret.shape[1]))  # Only positive, un-leveraged positions
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},)  # Weights must sum to 1
        results = minimize(fun=neg_sharpe,
                           x0=guess,
                           method='SLSQP',
                           options={'disp': False},
                           bounds=bounds,
                           constraints=constraints)
        return results

    def weights_for_return(self, ret_val, *, column='Close', industry='ALL', log_ret=None):
        if not isinstance(log_ret, pd.DataFrame) and not isinstance(self.__log_ret, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            # log_ret = np.log(df / df.shift(1))
            log_ret = df.pct_change()
        elif isinstance(self.__log_ret, pd.DataFrame):
            log_ret = self.__log_ret

        def min_vol(weights):
            return self._calc_ret_vol_sharpe(log_ret, weights)[1]  # Returns volatility given a set of weights

        guess = np.ones(log_ret.shape[1]) / log_ret.shape[1]  # Initial Guess of equal weights
        bounds = ((0, 1),) * log_ret.shape[1]  # Only positive, un-leveraged positions
        bounds = Bounds((0,)*log_ret.shape[1], (1,)*log_ret.shape[1])
        constraints = ({'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1},  # Ensures weights sum to 1
                       {'type': 'eq',
                        'fun': lambda weights: self._calc_ret_vol_sharpe(log_ret, weights)[0] - ret_val})  # Find desired return
        results = minimize(fun=min_vol,
                           x0=guess,
                           method='SLSQP',
                           options={'disp': True},
                           bounds=bounds,
                           constraints=constraints)
        return results.x

    def _sim_portfolios(self, log_ret, iters=5000):
        weights = np.zeros((iters, log_ret.shape[1]))
        rets = np.zeros(iters)
        vols = np.zeros(iters)
        sharpe = np.zeros(iters)

        for i in tqdm(range(iters), desc='Simulating Portfolios'):
            whole_weights = np.random.random(log_ret.shape[1])
            weights[i, :] = whole_weights / np.sum(whole_weights)

            rets[i], vols[i], sharpe[i] = self._calc_ret_vol_sharpe(log_ret, weights[i, :])

        return weights, rets, vols, sharpe

    @staticmethod
    def _calc_ret_vol_sharpe(log_ret, weights):
        ret = np.dot(log_ret.mean(), weights) * 252
        # ret = (np.sum(log_ret.mean() * weights) + 1) ** 252 - 1  # CHANGED
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov(), weights))) * np.sqrt(252)
        sharpe = ret / vol
        return ret, vol, sharpe
