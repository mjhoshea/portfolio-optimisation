import numpy as np

from BaseMarketEnv import BaseMarketEnv
from EfficientFrontier import EfficientFrontier


class GenerativeMarketEnv(BaseMarketEnv):

    def __init__(self, μ, Σ):
        super().__init__()
        self.μ = μ
        self.Σ = Σ
        self.best = np.argmax(μ)
        self.best_r = self.μ[self.best]
        self._ef = EfficientFrontier.from_sample_statistics(self.μ, self.Σ)

    def reset(self):
        pass

    def step(self, weights, eps_len=1, n_obs=1):
        returns = np.random.multivariate_normal(self.μ, self.Σ, (n_obs, eps_len))
        returns = np.mean(returns, axis=0)
        return returns*weights, returns

    def plot_efficient_frontier(self, allow_shorts=False, allow_lending=False, stds=None, returns=None, save_name=None):
        sr = self._ef.plot_frontier(allow_shorts=allow_shorts, allow_lending=allow_lending, stds = stds, returns = returns, save_name=save_name)
        return sr[0][0]

