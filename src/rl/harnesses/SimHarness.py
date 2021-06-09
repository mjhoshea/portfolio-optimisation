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
            rs = self.market.step(ws)
            avg_r = ws@rs
            regret = self._regret(rs, avg_r)

            self.trader.step(None, ws, avg_r, None, False)
            self.returns.append(avg_r)
            self.regrets.append(regret)

    def evaluate(self, n_episodes, render=False, render_sleep=0.25):
        pass

    @staticmethod
    def _regret(returns, avg_r):
        return max(returns) - avg_r

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

def plot_rewards(rewards, file_name=None):
    plt.plot(rewards)
    plt.title('Returns Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Return')
    if file_name:
        plt.savefig('./{}'.format(file_name))
        plt.clf()
    else:
        plt.show()