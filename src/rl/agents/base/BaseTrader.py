from abc import ABC, abstractmethod
import numpy as np


class BaseTrader(ABC):

    def __init__(self, n_assets):
        self.ϵ = 0.9
        self.ϵ_start = 0.99
        self.ϵ_end = 0.01
        self.ϵ_decay = 0.99
        self.ϵ_decay_steps = 1000
        self.timestep = 0
        self.start_ts = 1000
        self.n_assets = n_assets
        self.evaluate = False

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, is_done):
        pass

    def switch_to_evaluate(self):
        self.evaluate = True

    def switch_to_train(self):
        self.evaluate = False

    def decay_epsilon(self):
        self.ϵ = self.ϵ_end + (self.ϵ_start - self.ϵ_end) * \
                       np.exp(-1. * ((self.timestep - self.start_ts) / self.ϵ_decay_steps))
