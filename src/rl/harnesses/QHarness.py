import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class QHarness:

    def __init__(self, market, trader):
        self.market = market
        self.trader = trader
        self.n = len(self.trader.traders)
        self.returns = []
        self.regrets = []

    def train(self, n_episodes, n_t, verbose=False):
        self.trader.switch_to_training_mode()

        for episode in range(n_episodes):

            episode_returns = []
            episode_regrets = []
            self.trader.reset()
            ws = self.trader.w
            
            for t in range(n_t):
                next_ws, actions = self.trader.act(ws)

                returns, raw_returns = self.market.step(ws)
                avg_r = np.sum(returns)
                regret = self._regret(raw_returns, avg_r)

                self.trader.step(ws, actions, self.n*[avg_r], next_ws, None)

                ws = next_ws

                episode_returns.append(avg_r)
                episode_regrets.append(regret)

            self.returns.append(episode_returns)
            self.regrets.append(episode_regrets)

    def evaluate(self, n_episodes, render=False, render_sleep=0.25):
        pass

    def _regret(self, rs, avg_r):
        return rs[0][self.market.best] - avg_r

    def plot_training_results(self, file_name=None, window=None):
        returns = np.mean(self.returns, axis=0)
        std = np.std(self.returns, axis=0)
        if window:
            returns = np.array(pd.Series(returns).rolling(window).mean()[window - 1:])
            std = np.array(pd.Series(std).rolling(window).mean()[window - 1:])
        plot_rewards(returns, std=std, file_name=file_name)

    def plot_training_regret(self, file_name=None, window=None):
        regrets = np.mean(self.regrets, axis=0)
        std = np.std(self.regrets, axis=0)
        if window:
            regrets = np.array(pd.Series(regrets).rolling(window).mean()[window - 1:])
            std = np.array(pd.Series(std).rolling(window).mean()[window - 1:])
        plot_rewards(regrets, std, file_name)

def plot_rewards(r, std=None, file_name=None):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(r)
    plt.fill_between(range(len(r)), r + std, r - std, alpha=0.2)
    plt.title('Returns Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Return')
    if file_name:
        plt.savefig('./{}'.format(file_name))
        plt.clf()
    else:
        plt.show()