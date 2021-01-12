import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
from tqdm import tqdm


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.__ret_mat = None

    def plot_minimum_variance_frontier(self, column='Close', *, nonneg=True, leverage=1, industry='ALL', iters=5000,
                                       points=300, figsize=(12, 8), seed=42):
        np.random.seed(seed)
        if industry == 'ALL':
            df = self.df.pivot_table(values=column, index='Date', columns='ticker')
        else:
            df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')

        ret_mat = df.pct_change()
        ret_mat = ret_mat.dropna()
        self.__ret_mat = ret_mat

        weights, rets, vols, sharpe = self._sim_portfolios(ret_mat, iters)  # Simulates random portfolio weights

        ret_vol = rets[vols.argmin()]
        vol_vol = vols.min()

        sharpe_dict = self.max_sharpe_portfolio(nonneg=nonneg, leverage=leverage, ret_mat=ret_mat)
        ret_sharpe = sharpe_dict['return']
        vol_sharpe = sharpe_dict['risk']

        frontier_returns = np.linspace(ret_mat.mean().min(), ret_mat.mean().max(), points)

        frontier_vols = np.array([])
        for r in tqdm(frontier_returns, desc='Creating Frontier'):
            risk = self.weights_for_return(r, nonneg=nonneg, leverage=leverage, ret_mat=ret_mat)['risk']
            frontier_vols = np.append(frontier_vols, risk)

        frontier_returns = frontier_returns[~np.isnan(frontier_vols)]
        frontier_vols = frontier_vols[~np.isnan(frontier_vols)]
        frontier_returns = frontier_returns[frontier_vols > 0]
        frontier_vols = frontier_vols[frontier_vols > 0]

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

    def max_sharpe_portfolio(self, *, column='Close', nonneg=True, leverage=1, industry='ALL', ret_mat=None):
        if not isinstance(ret_mat, pd.DataFrame) and not isinstance(self.__ret_mat, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            ret_mat = df.pct_change()
            ret_mat = ret_mat.dropna()
        elif isinstance(self.__ret_mat, pd.DataFrame):
            ret_mat = self.__ret_mat

        mu = ret_mat.mean().values.reshape((1, ret_mat.shape[1]))
        sigma = ret_mat.cov().values

        w = cp.Variable(mu.shape[1])

        ret = cp.matmul(mu, w)
        risk = cp.quad_form(w, sigma)

        sharpe = -ret / cp.sqrt(risk)

        constraints = [
            cp.sum(w) == leverage
        ]

        if nonneg:
            constraints.append(w >= 0)
        else:
            constraints.append(w >= -1)

        problem = cp.Problem(cp.Minimize(sharpe), constraints=constraints)

        problem.solve()

        return {'sharpe': problem.value, 'return': ret.value, 'risk': np.sqrt(risk.value), 'weights': w.value}

    def weights_for_return(self, ret_val, *, nonneg=True, leverage=1, column='Close', industry='ALL', ret_mat=None):
        if not isinstance(ret_mat, pd.DataFrame) and not isinstance(self.__ret_mat, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            ret_mat = df.pct_change()
            ret_mat = ret_mat.dropna()
        elif isinstance(self.__ret_mat, pd.DataFrame):
            ret_mat = self.__ret_mat

        sigma = ret_mat.cov().values

        w = cp.Variable(sigma.shape[0])
        risk = cp.quad_form(w, sigma)

        constraints = [
            cp.sum(w) == leverage,
            cp.matmul(ret_mat.mean().values, w) == ret_val
        ]

        if nonneg:
            constraints.append(w >= 0)
        else:
            constraints.append(w >= -1)

        problem = cp.Problem(cp.Minimize(risk), constraints=constraints)
        problem.solve()

        return {'risk': np.sqrt(risk.value), 'weights': w.value}

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
        ret = np.dot(ret_mat.mean(), weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(ret_mat.cov(), weights)))
        sharpe = ret / vol
        return ret, vol, sharpe
