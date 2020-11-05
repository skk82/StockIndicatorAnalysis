from os import environ
import alpha_vantage as av

INDEX_TICKERS = {
    'JNJ', 'UNH', 'PFE', 'MRK', 'TMO', 'ABT',  # US Healthcare
    'ATVI', 'CMCSA', 'VZ', 'NFLX', 'DIS', 'T',  # US Communications
    'AAPL', 'MSFT', 'NVDA', 'ADBE', 'INTC', 'CSCO', 'PYPL',  # US Technology
    'AMZN', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT',  # US Consumer Discretionary
    'AMT', 'PLD', 'CCI', 'WY', 'AVB', 'PSA', 'SBAC', 'WELL',  # US Real Estate
    'CVX', 'XOM', 'KMI', 'WMB', 'COP', 'SLB', 'VLO', 'PXD',  # US Energy
    'PG', 'WMT', 'KO', 'PEP', 'COST', 'EL', 'CL', 'MO',  # US Consumer Staples
    'UNP', 'HON', 'UPS', 'MMM', 'CAT', 'LMT', 'RTX', 'BA',  # US Industrials
    'BRK.B', 'JPM', 'BAC', 'WFC', 'C', 'BLK', 'SPGI', 'MS',  # US Financials
    'NEE', 'DUK', 'D', 'SO', 'AEP', 'EXC', 'XEL', 'SRE',  # US Utilities
    'LIN', 'APD', 'SHW', 'NEM', 'ECL', 'DD', 'PPG', 'BLL',  # US Materials
    }


