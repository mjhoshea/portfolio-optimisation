import yfinance as yf

import numpy as np
import pandas as pd

from GenerativeMarketEnv import GenerativeMarketEnv
from HistoricalMarketEnv import HistoricalMarketEnv


class MarketFactory:

    def __init__(self, stocks, start_ymd, end_ymd, type='generative'):
        self.stocks = stocks
        self.start_ymd = start_ymd
        self.end_ymd = end_ymd
        self.type = type

    def create_market(self):
        stocks = ' '.join(self.stocks)
        yahoo_df = yf.download(stocks, start=self.start_ymd, end=self.end_ymd)
        returns_df = pd.DataFrame(columns=self.stocks)

        if self.type == 'generative':
            for stock in self.stocks:
                returns_df[stock] = yahoo_df['Close'][stock].pct_change()[1:]
            μ = np.array(returns_df.mean())
            Σ = np.array(returns_df.cov())
            return GenerativeMarketEnv(μ, Σ)
        elif self.type == 'historical':
            for stock in self.stocks:
                returns_df[stock] = (np.log(yahoo_df['Close'][stock]) - np.log(yahoo_df['Close'][stock].shift(1)))[1:]

            return HistoricalMarketEnv(len(self.stocks), returns_df, self.start_ymd, self.end_ymd)
        else:
            raise ValueError("{} i not a valid market type.".format(self.type))