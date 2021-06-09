import numpy as np
from abc import ABC

from BaseTrader import BaseTrader


class RandomTrader(BaseTrader, ABC):

    def __init__(self, n_assets):
        super().__init__(n_assets)

    def act(self, state):
        preference = np.random.uniform(0, 1, self.n_assets)
        normalised_weights = np.exp(preference) / sum(np.exp(preference))
        return normalised_weights

    def step(self, state, action, reward, next_state):
        pass
