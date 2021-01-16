# =============================================================================
# Backtesting strategy - I : Monthly portfolio rebalancing
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import datetime
import copy
import matplotlib.pyplot as plt


def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    n = len(df)/12
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["mon_ret"].std() * np.sqrt(12)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

# Download historical data (monthly) for DJI constituent stocks

tickers = ["MMM","AXP","T","BA","CAT","CVX","CSCO","KO", "XOM","GE","GS","HD",
           "IBM","INTC","JNJ","JPM","MCD","MRK","MSFT","NKE","PFE","PG","TRV",
           "UTX","UNH","VZ","V","WMT","DIS"]


INDEX_TICKERS = {
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'TMO', 'ABT'],  # US Healthcare
    'Communications': ['ATVI', 'CMCSA', 'VZ', 'NFLX', 'DIS', 'T'],  # US Communications
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'ADBE', 'INTC', 'CSCO', 'PYPL'],  # US Technology
    'Consumer Discretionary': ['AMZN', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT'],  # US Consumer Discretionary
    'Real Estate': ['AMT', 'PLD', 'CCI', 'WY', 'AVB', 'PSA', 'SBAC', 'WELL'],  # US Real Estate
    'Energy': ['CVX', 'XOM', 'KMI', 'WMB', 'COP', 'SLB', 'VLO', 'PXD'],  # US Energy
    'Consumer Staples': ['PG', 'WMT', 'KO', 'PEP', 'COST', 'EL', 'CL', 'MO'],  # US Consumer Staples
    'Industrials': ['UNP', 'HON', 'UPS', 'MMM', 'CAT', 'LMT', 'RTX', 'BA'],  # US Industrials
    'Financials': ['BRK-B', 'JPM', 'BAC', 'WFC', 'C', 'BLK', 'SPGI', 'MS'],  # US Financials
    'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'SRE'],  # US Utilities
    'Materials': ['LIN', 'APD', 'SHW', 'NEM', 'ECL', 'DD', 'PPG', 'BLL'],  # US Materials
    }

ohlc_mon = {}
for y in INDEX_TICKERS:
    for t in INDEX_TICKERS[y]:       
        alc = pd.read_csv('/Users/siddharthkantamneni/Documents/Grad School/project/Stock Info By Industry/'+y+'/'+t+'.csv')
        alc.set_index('Date', inplace=True)
        alc.index = pd.to_datetime(alc.index)
        alc = alc.resample('1M').mean()
        ohlc_mon[t] = alc

 # redefine tickers variable after removing any tickers with corrupted data

################################Backtesting####################################

# calculating monthly return for each stock and consolidating return info by stock in a separate dataframe
ohlc_dict = copy.deepcopy(ohlc_mon)
return_df = pd.DataFrame()
for y in INDEX_TICKERS:
    for t in INDEX_TICKERS[y]: 
        print("calculating monthly return for ",t)
        ohlc_dict[t]["mon_ret"] = ohlc_dict[t]["Adj Close"].pct_change()
        return_df[t] = ohlc_dict[t]["mon_ret"]


# function to calculate portfolio return iteratively
def pflio(DF,m,x):
    """Returns cumulative portfolio return
    DF = dataframe with monthly return info for all stocks
    m = number of stock in the portfolio
    x = number of underperforming stocks to be removed from portfolio monthly"""
    df = DF.copy()
    portfolio = []
    monthly_ret = [0]
    for i in range(1,len(df)):
        if len(portfolio) > 0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        portfolio = portfolio + new_picks
        #print(portfolio)
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns=["mon_ret"])
    return monthly_ret_df


#calculating overall strategy's KPIs
CAGR(pflio(return_df,6,3))
sharpe(pflio(return_df,6,3),0.025)
max_dd(pflio(return_df,6,3)) 

#calculating KPIs for Index buy and hold strategy over the same period
gspc =  wb.DataReader('^GSPC', data_source='yahoo', start='2000-01-01', end ='2020-11-18')
gspc.to_csv('/Users/siddharthkantamneni/Documents/Grad School/project/Stock Info By Industry/SnP.csv')
SnP = pd.read_csv('/Users/siddharthkantamneni/Documents/Grad School/project/Stock Info By Industry/SnP.csv')
SnP.set_index('Date', inplace=True)
SnP.index = pd.to_datetime(SnP.index)
SnP = SnP.resample('1M').mean()
SnP["mon_ret"] = SnP["Adj Close"].pct_change()
CAGR(SnP)
sharpe(SnP,0.025)
max_dd(SnP)

#visualization
fig, ax = plt.subplots()
plt.plot((1+pflio(return_df,25,3)[200:].reset_index(drop=True)).cumprod())
plt.plot((1+SnP["mon_ret"][201:].reset_index(drop=True)).cumprod())
plt.title("Index Return vs Strategy Return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["Strategy Return","Index Return"])