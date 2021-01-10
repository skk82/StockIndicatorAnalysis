from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

        log_ret = np.log(df / df.shift(1))
        self.__log_ret = log_ret

        weights, rets, vols, sharpe = self._sim_portfolios(log_ret, iters)
        max_ret = rets[sharpe.argmax()]
        max_vol = vols[sharpe.argmax()]

        frontier_returns = np.linspace(0, np.max(rets), points)

        with Pool(processes=cpu_count()-1) as pool:
            ret_weights = list(tqdm(pool.imap_unordered(self.weights_for_return, frontier_returns),
                                    desc='Creating Frontier', total=points))

        frontier_vols = [self._calc_ret_vol_sharpe(log_ret, w)[1] for w in ret_weights]

        plt.figure(figsize=figsize)
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(vols, rets, c=sharpe, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')

        plt.scatter(max_vol, max_ret, c='red', s=50)

        plt.plot(frontier_vols, frontier_returns, ':', color='orange', linewidth=2)
        plt.show()

    def max_sharpe_portfolio(self, *, column='Close', industry='ALL', log_ret=None):
        if not isinstance(log_ret, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            log_ret = np.log(df / df.shift(1))
        neg_sharpe = lambda weights: self._calc_ret_vol_sharpe(log_ret, weights)[2]
        guess = np.ones(log_ret.shape[1]) / log_ret.shape[1]
        bounds = ((0, 1),) * log_ret.shape[1]
        constraints = ({'type': 'eq', 'fun': lambda weights: 1 - np.sum(weights)},)
        results = minimize(neg_sharpe, guess, method='SLSQP', options={'disp': False}, bounds=bounds,
                           constraints=constraints)
        return results

    def weights_for_return(self, ret_val, *, column='Close', industry='ALL', log_ret=None):
        if not isinstance(log_ret, pd.DataFrame) and not isinstance(self.__log_ret, pd.DataFrame):
            if industry == 'ALL':
                df = self.df.pivot_table(values=column, index='Date', columns='ticker')
            else:
                df = self.df[self.df.industry == industry].pivot_table(values=column, index='Date', columns='ticker')
            log_ret = np.log(df / df.shift(1))
        elif isinstance(self.__log_ret, pd.DataFrame):
            log_ret = self.__log_ret
        min_vol = lambda weights: self._calc_ret_vol_sharpe(log_ret, weights)[1]
        guess = np.ones(log_ret.shape[1]) / log_ret.shape[1]
        bounds = ((0, 1),) * log_ret.shape[1]
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

        for i in tqdm(range(iters), desc='Simulating Portfolios'):
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
