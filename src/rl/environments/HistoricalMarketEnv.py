import numpy as np
from dateutil import relativedelta, parser
import datetime

from BaseMarketEnv import BaseMarketEnv


class HistoricalMarketEnv(BaseMarketEnv):

    def __init__(self, n_assets, returns_df, train_start_date, train_end_date):
        super().__init__()
        self.n_assets = n_assets
        self.train_start_date = parser.parse(train_start_date)
        self.train_end_date = self.train_start_date + relativedelta.relativedelta(months=6)
        self.returns_df = returns_df

    def step(self, weights, eps_len=1, n_obs=1):
        mask = \
            (self.returns_df.index > str(self.train_start_date)) & (self.returns_df.index <= str(self.train_end_date))
        train = self.returns_df[mask]
        start = np.random.randint(low=0, high=len(train) - eps_len)
        returns = train[start:start + eps_len]
        return returns * weights

    def get_test_data(self):
        test_end_date = self.train_end_date + relativedelta.relativedelta(months=1)
        mask = \
            (self.returns_df.index > str(self.train_end_date)) & (self.returns_df.index <= str(test_end_date))
        return self.returns_df[mask]

    def slide_window(self):
        self.train_start_date += relativedelta.relativedelta(months=1)
        self.train_end_date += relativedelta.relativedelta(months=1)

    def reset(self):
        pass
