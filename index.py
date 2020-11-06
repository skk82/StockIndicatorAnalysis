from os import environ

import pandas as pd
from datetime import date

from alpha_vantage.timeseries import TimeSeries

KEY = environ['AVKey']

INDEX_TICKERS = {
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'TMO', 'ABT'],  # US Healthcare
    'Communications': ['ATVI', 'CMCSA', 'VZ', 'NFLX', 'DIS', 'T'],  # US Communications
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'ADBE', 'INTC', 'CSCO', 'PYPL'],  # US Technology
    'Consumer Discretionary': ['AMZN', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT'],  # US Consumer Discretionary
    'Real Estate': ['AMT', 'PLD', 'CCI', 'WY', 'AVB', 'PSA', 'SBAC', 'WELL'],  # US Real Estate
    'Energy': ['CVX', 'XOM', 'KMI', 'WMB', 'COP', 'SLB', 'VLO', 'PXD'],  # US Energy
    'Consumer Staples': ['PG', 'WMT', 'KO', 'PEP', 'COST', 'EL', 'CL', 'MO'],  # US Consumer Staples
    'Industrials': ['UNP', 'HON', 'UPS', 'MMM', 'CAT', 'LMT', 'RTX', 'BA'],  # US Industrials
    'Financials': ['BRK.B', 'JPM', 'BAC', 'WFC', 'C', 'BLK', 'SPGI', 'MS'],  # US Financials
    'Utilities': ['NEE', 'DUK', 'D', 'SO', 'AEP', 'EXC', 'XEL', 'SRE'],  # US Utilities
    'Materials': ['LIN', 'APD', 'SHW', 'NEM', 'ECL', 'DD', 'PPG', 'BLL'],  # US Materials
    }


def get_index(start: date, end: date, interval: str = '1d', long: bool = True, path: str = './',
              **kwargs) -> None:
    """

    :param start: Date object, the earliest date to retrieve data for. Format: YYYY/MM/DD
    :param end: Date object, the latest date to retrieve data for. Format: YYYY/MM/DD
    :param interval: The period between stock data retrievals choices: 1min, 5min, 15min, 30min, 60min, 1d, 1w, 1m
    :param long: True retrieves all data, False retrieves the 100 most recent entries
    :param path: A path to a directory to save the file
    """
    path = path + f'{start.year}.{start.month}.{start.day}_{end.year}.{end.month}.{end.day}_{interval}.csv'

    if long:
        outputsize = 'full'
    else:
        outputsize = 'compact'

    ts = TimeSeries(key=KEY, output_format='pandas')

    df = pd.DataFrame()

    if interval in ['1min', '5min', '15min', '30min', '60min']:
        for sector, tickers in INDEX_TICKERS.items():
            for tick in tickers:
                df_tick, _ = ts.get_intraday(symbol=tick, interval=interval, outputsize=outputsize)
                df_tick['sector'], df_tick['symbol'] = sector, tick
                df_tick = df_tick[(df_tick.index >= start) & (df_tick.index <= end)]
                df_tick = df_tick.reset_index()
                df_tick = df_tick.rename({'index': 'date'}, axis=1)
                df = df.append(df_tick)

    elif interval == '1d':
        for sector, tickers in INDEX_TICKERS.items():
            for tick in tickers:
                df_tick, _ = ts.get_daily(symbol=tick, interval=interval, outputsize=outputsize)
                df_tick['sector'], df_tick['symbol'] = sector, tick
                df_tick = df_tick[(df_tick.index >= start) & (df_tick.index <= end)]
                df_tick = df_tick.reset_index()
                df_tick = df_tick.rename({'index': 'date'}, axis=1)
                df = df.append(df_tick)

    elif interval == '1w':
        for sector, tickers in INDEX_TICKERS.items():
            for tick in tickers:
                df_tick, _ = ts.get_weekly(symbol=tick, interval=interval, outputsize=outputsize)
                df_tick['sector'], df_tick['symbol'] = sector, tick
                df_tick = df_tick[(df_tick.index >= start) & (df_tick.index <= end)]
                df_tick = df_tick.reset_index()
                df_tick = df_tick.rename({'index': 'date'}, axis=1)
                df = df.append(df_tick)

    elif interval == '1m':
        for sector, tickers in INDEX_TICKERS.items():
            for tick in tickers:
                df_tick, _ = ts.get_monthly(symbol=tick, interval=interval, outputsize=outputsize)
                df_tick['sector'], df_tick['symbol'] = sector, tick
                df_tick = df_tick[(df_tick.index >= start) & (df_tick.index <= end)]
                df_tick = df_tick.reset_index()
                df_tick = df_tick.rename({'index': 'date'}, axis=1)
                df = df.append(df_tick)

    df.columns = [s.split(' ')[-1] for s in df.columns]
    df = df[df.index[-2:]+df.index[:-2]]

    print('Data Retrieved!')
    print(f'Start: {start}, End: {end}, Interval: {interval}')
    print(f'File stored at: {path}')

    df.to_csv(path, index=False)


if __name__ == '__main__':
    from sys import argv
    from dateutil import relativedelta
    try:  # TODO: Need to parse dates
        params = dict(arg.split('=') for arg in argv)
        if 'end' not in params:
            params['end'] = date.today()
        else:
            params['end'] = date.pa
        if 'start' not in params:
            params['start'] = params['end'] - relativedelta(years=2)
        get_index(**params)
    except:
        print(argv)
        help(get_index)
