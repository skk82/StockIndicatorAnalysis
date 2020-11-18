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


for y in INDEX_TICKERS:
    for t in INDEX_TICKERS[y]:
        x = wb.DataReader(t, data_source='yahoo', start='2000-01-01')
        x.to_csv('/Users/siddharthkantamneni/Documents/Grad School/project/'+t+'.csv')