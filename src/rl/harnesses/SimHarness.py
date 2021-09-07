import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SimHarness:

    def __init__(self, market, trader):
        self.market = market
        self.trader = trader
        self.returns = []
        self.regrets = []

    def train(self, n_episodes, verbose=False):
        for episode in range(n_episodes):
            ws = self.trader.act(None)
            returns, raw_returns = self.market.step(ws)
            avg_r = np.sum(returns)
            regret = self._regret(raw_returns, avg_r)

            self.trader.step(None, ws, avg_r, None, False)
            self.returns.append(avg_r)
            self.regrets.append(regret)

    def evaluate(self, n_episodes, render=False, render_sleep=0.25):
        pass

    def _regret(self, rs, avg_r):
        return rs[0][self.market.best] - avg_r

    def plot_training_results(self, file_name=None, window=None):
        returns = self.returns
        if window:
            returns = np.array(pd.Series(self.returns).rolling(window).mean()[window-1:])
        plot_rewards(returns, file_name)

    def plot_training_regret(self, file_name=None, window=None):
        regrets = self.regrets
        if window:
            regrets = np.array(pd.Series(self.regrets).rolling(window).mean()[window-1:])
        plot_rewards(regrets, file_name)

    def plot_cum_regret(self, file_name=None, window=None):
        cum_regrets = np.cumsum(self.regrets)
        if window:
            cum_regrets = np.array(pd.Series(self.regrets).rolling(window).mean()[window-1:])
        plot_rewards(cum_regrets, file_name)


def plot_rewards(rewards, file_name=None):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(rewards)
    plt.xlabel('Timestep')
    plt.ylabel('Return')
    if file_name:
        plt.savefig('./{}'.format(file_name))
        plt.clf()
    else:
        plt.show()