from BaseTrader import BaseTrader

import itertools
import numpy as np


class TabularTrader(BaseTrader):

    def __init__(self, args):
        super().__init__(args.n_assets)
        self.ϵ = args.ϵ
        self.n_inc = args.n_inc
        self.actions = self._calculate_valid_weights()
        self._n_viable_portfolios = len(self.actions)
        self.Q = None
        self.N = None
        self.t = None
        self.reset()

    def act(self, state):
        ϵ = self.ϵ if not callable(self.ϵ) else self.ϵ(self.t)
        π = np.ones(self._n_viable_portfolios) * (ϵ / self._n_viable_portfolios)
        greedy_action = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
        π[greedy_action] += (1.0 - ϵ)
        i = np.random.choice(np.arange(self._n_viable_portfolios), p=π)
        return self.actions[i]

    def step(self, state, action, reward, next_state, is_done):
        i = self.actions.index(action)
        self.N[i] += 1
        self.Q[i] += (reward - self.Q[i]) / self.N[i]

    def reset(self):
        self.Q = np.zeros(self._n_viable_portfolios)
        self.N = np.zeros(self._n_viable_portfolios)
        self.t = 1

    def _calculate_valid_weights(self):
        ws = []
        for n in range(self.n_assets):
            ws.append(np.linspace(0, 1, self.n_inc))
        ws = list(itertools.product(*ws))
        return list(filter(lambda a: sum(a) == 1, ws))
