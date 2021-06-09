from abc import ABC, abstractmethod


class BaseMarketEnv(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def step(self, n_steps):
        pass

    @abstractmethod
    def reset(self):
        pass