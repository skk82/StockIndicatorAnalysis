import numpy as np
import pandas as pd
import cvxopt as cp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

from analysis.portfolio_functions.Errors.Errors import InfeasibleError


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.__ret_mat = None

    def plot_minimum_variance_frontier(self, column='Close', *, nonneg=True, leverage=1, industry='ALL',
                                       points=300, figsize=(12, 8), seed=42):
        np.random.seed(seed)
        if industry == 'ALL':
            df = self.df.pivot_table(values=column, index='Date', columns='ticker')
        else:
            df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')

        ret_mat = df.pct_change()
        ret_mat = ret_mat.dropna()
        self.__ret_mat = ret_mat

        frontier_returns = np.linspace(ret_mat.mean().min(), ret_mat.mean().max()*leverage, points)

        frontier_vols = np.array([])
        bad_indices = []

        for r in tqdm(frontier_returns, desc='Creating Frontier'):
            try:
                risk = self.weights_for_return(r, nonneg=nonneg, leverage=leverage, ret_mat=ret_mat)['risk']
                frontier_vols = np.append(frontier_vols, risk)
            except InfeasibleError:
                bad_indices.append(np.argwhere(frontier_returns == r))

        # Cleaning frontier returns
        frontier_returns = np.delete(frontier_returns, bad_indices)
        frontier_returns = frontier_returns[~np.isnan(frontier_vols)]
        frontier_vols = frontier_vols[~np.isnan(frontier_vols)]
        frontier_returns = frontier_returns[frontier_vols > 0]
        frontier_vols = frontier_vols[frontier_vols > 0]

        # Finding max Sharpe ratio portfolio
        frontier_sharpe = frontier_returns / frontier_vols * 252 / np.sqrt(252)
        ret_sharpe = frontier_returns[frontier_sharpe.argmax()]
        vol_sharpe = frontier_vols[frontier_sharpe.argmax()]

        # Finding min volatility portfolio
        ret_vol = frontier_returns[frontier_vols.argmin()]
        vol_vol = frontier_vols.min()

        # Formatting figure
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Setting labels
        ax.set_title('Minimum Variance Frontier')
        ax.set_xlabel('Daily Volatility')
        ax.set_ylabel('Daily Return')

        # Plotting frontier with gradient for Sharpe ratio
        frontier_points = np.array([frontier_vols, frontier_returns]).T.reshape(-1, 1, 2)
        segments = np.concatenate([frontier_points[:-1], frontier_points[1:]], axis=1)
        norm = plt.Normalize(frontier_sharpe.min(), frontier_sharpe.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(frontier_sharpe)
        lc.set_linewidth(4)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

        # Plotting point for maximum Sharpe ratio
        ax.scatter(vol_sharpe, ret_sharpe, marker='*', c='lime', s=500,
                   label=f'Maximum Sharpe Ratio: {frontier_sharpe.max():.2f}')

        # Plotting point for minimum variance portfolio
        ax.scatter(vol_vol, ret_vol, marker='*', c='orange', s=500, label='Minimum Variance Portfolio')

        ax.legend(labelspacing=0.5, loc=2)
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

        sharpe = rets / vols * 252 / np.sqrt(252)

        return {'sharpe': sharpe.max(), 'daily return': rets[sharpe.argmax()], 'daily risk': vols[sharpe.argmax()],
                'weights': weights[sharpe.argmax()]}

    def weights_for_return(self, ret_val, *, column='Close', nonneg=True, leverage=1, industry='ALL', ret_mat=None):
        if not isinstance(ret_mat, pd.DataFrame) and not isinstance(self.__ret_mat, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            ret_mat = df.pct_change()
            ret_mat = ret_mat.dropna()
        elif isinstance(self.__ret_mat, pd.DataFrame):
            ret_mat = self.__ret_mat

        sigma = cp.matrix(ret_mat.cov().values)  # P

        null = cp.spmatrix([], [], [], (ret_mat.shape[1], 1))  # q: a matrix of zeros


        # TODO: include bounds on the weights of each stock
        ineq_const = cp.matrix(-ret_mat.mean().values)  # G

        ret_val = cp.matrix(ret_val)  # h

        eq_const = cp.matrix(1, (1, 4))


        # TODO: Should this be an inequality?
        one = cp.matrix(leverage)


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

        if problem.status == 'infeasible':
            raise InfeasibleError

        return {'risk': np.sqrt(risk.value), 'weights': weights.value}
