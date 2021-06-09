import numpy as np
from scipy.special import softmax

from QSingleAssetTrader import QSingleAssetTrader


class QTrader:

    def __init__(self, args):
        self.args = args
        self.n_assets = args.num_assets
        self.increment = args.increment
        self.w = softmax(np.ones(self.n_assets))
        self.traders = self._initialise_traders(args)

    def act(self, w):
        actions = [trader.act(w[i]) for i, trader in enumerate(self.traders)]
        Δw = (np.array(actions) - 1)*self.increment
        self.w += Δw
        return softmax(self.w), actions

    def step(self, ws, actions, avr_r, next_ws, done):
        for i, trader in enumerate(self.traders):
            trader.step(ws[i], actions[i], avr_r[i], next_ws[i], False)

    def reset(self):
        self.w = softmax(np.ones(self.n_assets))
        self.traders = self._initialise_traders(self.args)

    def switch_to_training_mode(self):
        [trader.switch_to_train() for trader in self.traders]

    def _initialise_traders(self, args):
        return [QSingleAssetTrader(1, 3, args) for _ in range(self.n_assets)]