import pandas as pd
import matplotlib.pyplot as plt


class MinimumVariancePortfolio:

    def __init__(self, df: pd.DataFrame):
        self.df = df
