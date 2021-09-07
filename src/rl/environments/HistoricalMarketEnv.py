import numpy as np

from BaseMarketEnv import BaseMarketEnv


class HistoricalMarketEnv(BaseMarketEnv):

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def step(self, weights, eps_len=1, n_obs=1):
        start = np.random.randint(low=0,high=len(self.train) - eps_len)
        returns = self.train[start:start+eps_len]
        return returns*weights

    def reset(self):
        pass
