import numpy as np

from BaseMarketEnv import BaseMarketEnv


class GenerativeMarketEnv(BaseMarketEnv):

    def __init__(self, μ, Σ):
        super().__init__()
        self.μ = μ
        self.Σ = Σ

    def reset(self):
        pass

    def step(self, weights):
        returns = np.random.multivariate_normal(self.μ, self.Σ)
        return returns
