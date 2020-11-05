import pandas as pd

# TODO: Check historical data

INDEX_TICKERS = {
    'JNJ', 'UNH', 'PFE', 'MRK', 'TMO', 'ABT',  # US Healthcare
    'FB', 'GOOGL', 'GOOG', 'NFLX', 'TMUS', 'CHTR',  # US Communications
    'AAPL', 'MSFT', 'NVDA', 'V', 'MA', 'CRM', 'PYPL',  # US Technology
    'AMZN', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT',  # US Consumer Discretionary
    'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'SBAC', 'WELL',  # US Real Estate
    'CVX', 'XOM', 'KMI', 'WMB', 'COP', 'SLB', 'MPC', 'PSX',  # US Energy
    'PG', 'WMT', 'KO', 'PEP', 'COST', 'MDLZ', 'CL', 'MO',  # US Consumer Staples
    'UNP', 'HON', 'UPS', 'MMM', 'CAT', 'LMT', 'RTX', 'BA',  # US Industrials
    'BRK.B', 'JPM', 'BAC', 'WFC', 'C', 'BLK', 'SPGI', 'MS',  # US Financials
    'NEE', 'DUK', 'D', 'SO', 'AEP', 'EXC', 'XEL', 'SRE',  # US Utilities
    'LIN', 'APD', 'SHW', 'NEM', 'ECL', 'DD', 'PPG', 'DOW',  # US Materials
    }
