# Stock Indicator Analysis
Siddharth Kantamneni (skk82@cornell.edu)

Kyle Walsh (kew96@cornell.edu)

## Project Goals
While this is an ongoing research project, we are implementing some of the most common equity models and exploring 
their derivations. We are hoping to apply theoretical concepts that we are learning at university to real world 
problems using modern data.

## Retrieving Data

In order to retrieve data, we recommend using the file stock_script.py. This can be run from the command line using the
pattern: `python stock_script.py end=<YYYY-MM-dd> start=<YYYY-MM-dd> tickers=<tickers> path=<path>`. All parameters are 
optional with `end` and
`start` defaulting to today and `end` minus two years respectively. Furthermore, a `path` can be specified to save the
resulting files in a specific location where the default is a directory called indices in the root directory. Lastly,
`tickers` can be specified using either a dictionary or path to a txt file containing a dictionary following the
format: `{'<industry 1>': ['<ticker 1>', '<ticker 2>', ...], '<industry 2>': ['<ticker 3>', '<ticker 4>'], ...}`.

## Next Steps

- [ ] Include documentation
- [ ] Add theory to notebooks
- [ ] Quadratic Programming
  - [ ] Incorporate different lending and borrowing rates
  - [ ] Adjust constraints to be greater than or equal to desired return
  - [ ] Consider cases where risk free rate is greater than return of minimum variance portfolio
- [ ] Begin Fama-French Three Factor Model
