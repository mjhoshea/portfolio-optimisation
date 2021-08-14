from abc import ABC, abstractmethod


class BaseMarketEnv(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def step(self, n_steps, n=1):
        pass

    @abstractmethod
    def reset(self):
        pass