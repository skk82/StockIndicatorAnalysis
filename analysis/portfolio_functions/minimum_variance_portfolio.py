from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from cvxopt import matrix, solvers
from tqdm import tqdm


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.__ret_mat = None

    def plot_minimum_variance_frontier(self, column='Close', *, industry='ALL', iters=5000, points=300,
                                       figsize=(12, 8), seed=42):
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

        frontier_returns = np.linspace(1.05 * rets.min(), 0.95 * rets.max(), points)

        with Pool(processes=min((cpu_count() - 1), 8)) as pool:  # Multiprocessing to speed up frontier calculation
            results = list(tqdm(pool.imap_unordered(self.weights_for_return, frontier_returns),
                                desc='Creating Frontier', total=points))

        frontier_weights = [np.array(r['x']) for r in results]
        frontier_vols = []
        i = 0
        for w in frontier_weights:
            r, v, s = self._calc_ret_vol_sharpe(ret_mat, w.flatten())
            if i < 5:
                print(w.flatten())
                print(v, '\n\n')
            i += 1
            frontier_vols.append(v)

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

        initvals = {'x': matrix(np.zeros(ret_mat.shape[1]) / ret_mat.shape[1], tc='d')}

        P = 2*matrix(ret_mat.cov().values, tc='d')  # Covariance matrix of returns
        q = matrix(np.zeros((ret_mat.shape[1], 1)), tc='d')

        # Negative identity matrix times each weight must be less than zero, non-negative constraint
        G = matrix(np.diag([-1 for _ in range(ret_mat.shape[1])]), tc='d')
        h = matrix(np.zeros(ret_mat.shape[1]), tc='d')

        # The sum of the weights must be 1 and the weighted average of returns must be the desired ret_val
        A = matrix(np.vstack([np.ones((1, ret_mat.shape[1])), ret_mat.mean().values]), tc='d')
        b = matrix([1.0, ret_val], tc='d')

        results = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b, initvals=initvals)
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
