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

        frontier_returns = np.linspace(ret_mat.mean().min(), ret_mat.mean().max(), points)

        frontier_vols = np.array([])
        for r in tqdm(frontier_returns, desc='Creating Frontier'):
            risk = self.weights_for_return(r, nonneg=nonneg, leverage=leverage, ret_mat=ret_mat)['risk']
            frontier_vols = np.append(frontier_vols, risk)

        frontier_returns = frontier_returns[~np.isnan(frontier_vols)]
        frontier_vols = frontier_vols[~np.isnan(frontier_vols)]
        frontier_returns = frontier_returns[frontier_vols > 0]
        frontier_vols = frontier_vols[frontier_vols > 0]

        frontier_sharpe = frontier_returns / frontier_vols
        ret_sharpe = frontier_returns[frontier_sharpe.argmax()]
        vol_sharpe = frontier_vols[frontier_sharpe.argmax()]

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=figsize)
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.plot(frontier_vols, frontier_returns, ':', color='red', linewidth=3, zorder=2)

        plt.scatter(vols, rets, c=sharpe, cmap='YlGnBu', s=50, alpha=0.3, zorder=1)
        plt.colorbar(label='Sharpe Ratio')

        plt.scatter(vol_sharpe, ret_sharpe, marker='*', c='lime', s=500, label=f'Maximum Sharpe Ratio '
                                                                               f'{frontier_sharpe.max()}', zorder=3)
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

        rets = np.linspace(ret_mat.mean().min(), ret_mat.mean().max(), ret_mat.shape[1]*200)

        vols = np.array([])
        for r in tqdm(rets, desc='Calculating Sharpe'):
            result = self.weights_for_return(r, nonneg=nonneg, leverage=leverage, ret_mat=ret_mat)['risk']
            risk = result['risk']
            weights = result['weights']
            vols = np.append(vols, risk)

        rets = rets[~np.isnan(vols)]
        weights = weights[~np.isnan(vols)]
        vols = vols[~np.isnan(vols)]
        rets = rets[vols > 0]
        weights = weights[vols > 0]
        vols = vols[vols > 0]

        sharpe = rets / vols

        return {'sharpe': sharpe.max(), 'return': rets[sharpe.argmax()], 'risk': vols[sharpe.argmax()],
                'weights': weights[sharpe.argmax()]}

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

        weights = cp.Variable(sigma.shape[0])
        risk = cp.quad_form(weights, sigma)

        constraints = [
            cp.sum(weights) == leverage,
            cp.matmul(ret_mat.mean().values, weights) == ret_val
        ]

        if nonneg:
            constraints.append(weights >= 0)
        else:
            constraints.append(weights >= -1)

        problem = cp.Problem(cp.Minimize(risk), constraints=constraints)
        problem.solve()

        return {'risk': np.sqrt(risk.value), 'weights': weights.value}

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
