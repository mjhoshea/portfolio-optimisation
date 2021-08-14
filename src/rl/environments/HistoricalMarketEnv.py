from BaseMarketEnv import BaseMarketEnv


class HistoricalMarketEnv(BaseMarketEnv):

    def __init__(self, hist):
        self.hist = hist

    def step(self, n_steps):
        pass

    def reset(self):
        pass
