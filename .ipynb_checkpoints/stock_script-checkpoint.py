import os

from tqdm import tqdm
from pandas_datareader import data as wb

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


def get_index(start: str, end: str, tickers: [dict, str], path: str = './indices/', **kwargs) -> None:
    """
    Retrieves pricing data for stocks and saves the prices to individual csv files. Can be run via command line
    with arguments passed as parameter=value

    :param start: A string of format YYYY-MM-dd for the first date to pull data from
    :param end: A string of format YYYY-MM-dd for the last date to pull data from
    :param tickers: Either a dictionary or a txt file containing a dictionary with industries as the keys and a list
    of tickers as the values
    :param path: The path to the directory to save all of csv files
    :rtype: Returns nothing. Saves csv files containing daily prices to a specified directory
    """

    if not os.path.exists(path):
        os.makedirs(path)

    if type(tickers) == str:
        import json
        with open(tickers) as file:
            data = file.read()
        tickers = json.loads(data)

    for industry in tqdm(tickers, desc=' Industries', position=0):
        for tick in tqdm(tickers[industry], desc=f' {industry}', position=1, leave=False):
            data = wb.DataReader(tick, data_source='yahoo', start=start, end=end)
            data.insert(0, 'industry', industry)
            full_path = path+f"{tick}_{start.split('-')[0]}.{start.split('-')[1]}.{start.split('-')[2]}_{end.split('-')[0]}.{end.split('-')[1]}.{end.split('-')[2]}.csv"
            data.to_csv(full_path)

    print('Data Retrieved!')
    print(f'Start: {start}, End: {end}')
    print(f'Data stored in folder: {path}')
    full_path = f"<ticker>_{start.split('-')[0]}.{start.split('-')[1]}.{start.split('-')[2]}_{end.split('-')[0]}.{end.split('-')[1]}.{end.split('-')[2]}.csv"
    print(f"File formats: {full_path}")


if __name__ == '__main__':
    from sys import argv
    from datetime import date
    from datetime import datetime
    from dateutil import relativedelta
    try:
        params = dict(arg.split('=') for arg in argv[1:])
        if 'end' not in params:
            params['end'] = date.today().strftime(format='%Y-%m-%d')
        if 'start' not in params:
            print(datetime.strptime(params['end'], '%Y-%m-%d'))
            print((datetime.strptime(params['end'], '%Y-%m-%d') - relativedelta(years=2)))
            print((datetime.strptime(params['end'], '%Y-%m-%d') - relativedelta(years=2)).date())
            params['start'] = (datetime.strptime(params['end'], '%Y-%m-%d')-relativedelta(years=2)).date().strftime(
                     format='%Y-%m-%d')
        if 'tickers' not in params:
            params['tickers'] = INDEX_TICKERS
        get_index(**params)
    except KeyError as e:
        print(e, '\n')
        print(argv, '\n')
        help(get_index)

    except ValueError as e:
        print(e, '\n')
        print(argv, '\n')
        help(get_index)
