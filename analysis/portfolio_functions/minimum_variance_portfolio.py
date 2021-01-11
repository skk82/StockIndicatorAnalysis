from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from tqdm import tqdm


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.__ret_mat = None

    def plot_minimum_variance_frontier(self, column='Close', *, industry='ALL', iters=5000, points=300,
                                       figsize=(12, 8), seed=42):
        assert points == 0 or points >= 300, '"points" must be 0 or greater than 300 to ensure that any curve plotted' \
                                             'is smooth'
        np.random.seed(seed)
        if industry == 'ALL':
            df = self.df.pivot_table(values=column, index='Date', columns='ticker')
        else:
            df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')

        ret_mat = df.pct_change()
        self.__ret_mat = ret_mat

        weights, rets, vols, sharpe = self._sim_portfolios(ret_mat, iters)  # Simulates random portfolio weights

        ret_vol = rets[vols.argmin()]
        vol_vol = vols.min()

        ret_sharpe, vol_sharpe, _ = self._calc_ret_vol_sharpe(ret_mat, self.max_sharpe_portfolio().x)

        frontier_returns = np.linspace(1.05*rets.min(), 0.95*rets.max(), points)

        with Pool(processes=cpu_count() - 1) as pool:  # Multiprocessing to speed up frontier calculation
            ret_weights = list(tqdm(pool.imap_unordered(self.weights_for_return, frontier_returns),
                                    desc='Creating Frontier', total=points))

        frontier_vols = [w.fun for w in ret_weights]

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=figsize)
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.plot(frontier_vols, frontier_returns, ':', color='red', linewidth=3, zorder=2)

        plt.scatter(vols, rets, c=sharpe, cmap='YlGnBu', s=50, alpha=0.3, zorder=1)
        plt.colorbar(label='Sharpe Ratio')

        plt.scatter(vol_sharpe, ret_sharpe, marker='*', c='lime', s=500, label='Maximum Sharpe Ratio', zorder=3)
        plt.scatter(vol_vol, ret_vol, marker='*', c='orange', s=500, label='Minimum Variance Portfolio', zorder=4)

        plt.legend(labelspacing=0.5, loc='best')
        plt.show()

    def max_sharpe_portfolio(self, *, column='Close', industry='ALL', ret_mat=None):
        if not isinstance(ret_mat, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            ret_mat = df.pct_change()

        def neg_sharpe(weights):
            return -self._calc_ret_vol_sharpe(ret_mat, weights)[2]  # Maximize Sharpe by minimizing negative Sharpe

        guess = np.ones(ret_mat.shape[1]) / ret_mat.shape[1]  # Initial guess of equal weights
        bounds = tuple((0, 1) for _ in range(ret_mat.shape[1]))  # Only positive, un-leveraged positions
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},)  # Weights must sum to 1
        results = minimize(fun=neg_sharpe,
                           x0=guess,
                           method='SLSQP',
                           options={'disp': False},
                           bounds=bounds,
                           constraints=constraints)
        return results

    def weights_for_return(self, ret_val, *, column='Close', industry='ALL', ret_mat=None):
        if not isinstance(ret_mat, pd.DataFrame) and not isinstance(self.__ret_mat, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            ret_mat = df.pct_change()
        elif isinstance(self.__ret_mat, pd.DataFrame):
            ret_mat = self.__ret_mat

        def min_vol(weights):
            return self._calc_ret_vol_sharpe(ret_mat, weights)[1]  # Returns volatility given a set of weights

        def port_return(weights):
            return self._calc_ret_vol_sharpe(ret_mat, weights)[0]

        guess = np.ones(ret_mat.shape[1]) / ret_mat.shape[1]  # Initial Guess of equal weights
        bounds = tuple((0, 1) for _ in range(ret_mat.shape[1]))  # Only positive, un-leveraged positions
        constraints = ({'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1},  # Ensures weights sum to 1
                       {'type': 'eq',
                        'fun': lambda weights: port_return(weights) - ret_val})  # Find desired return
        results = minimize(fun=min_vol,
                           x0=guess,
                           method='SLSQP',
                           options={'disp': True},
                           bounds=bounds,
                           constraints=constraints)
        return results

    def _sim_portfolios(self, ret_mat, iters=5000):
        weights = np.zeros((iters, ret_mat.shape[1]))  # Vector for weights
        rets = np.zeros(iters)  # Vector for returns
        vols = np.zeros(iters)  # Vector for volatilities
        sharpe = np.zeros(iters)  # Vector for Sharpe ratios

        for i in tqdm(range(iters), desc='Simulating Portfolios'):
            # Randomly generate weights and normalize between 0 and 1
            whole_weights = np.random.random(ret_mat.shape[1])
            weights[i, :] = whole_weights / np.sum(whole_weights)

            rets[i], vols[i], sharpe[i] = self._calc_ret_vol_sharpe(ret_mat, weights[i, :])

        return weights, rets, vols, sharpe

    @staticmethod
    def _calc_ret_vol_sharpe(ret_mat, weights):
        # Annualized portfolio return and volatility
        ret = np.dot(ret_mat.mean(), weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(ret_mat.cov(), weights))) * np.sqrt(252)
        sharpe = ret / vol
        return ret, vol, sharpe
