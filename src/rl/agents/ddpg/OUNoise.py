import copy
import random

import numpy as np


class OUNoise:
    def __init__(self, size, μ=0.0, θ=0.15, σ=0.2):
        self.state = np.float64(0.0)
        self.μ = μ * np.ones(size)
        self.θ = θ
        self.σ = σ
        self.reset()

    def reset(self):
        self.state = copy.copy(self.μ)

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.θ * (self.μ - x) + self.σ * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state
