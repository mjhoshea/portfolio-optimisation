import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import softmax


class HistoricalGaussianHarness:

    def __init__(self, env, policy, episode_length=50, reward_mode='returns', η=0.05, obs_period=1):
        self._env = env
        self._policy = policy
        self._episode_length = episode_length
        # self._optimal_val = max(env.μ) * episode_length

        self._obs_period = 1

        # differential sharpe ratio params
        self._reward_mode = reward_mode
        self._At = 0
        self._Atm1 = 0
        self._Bt = 0
        self._Btm1 = 0
        self._Dt = 0
        self._η = η

        self.test_dates = []
        self.historical_policy_returns = []
        self.historical_eq_weight_returns = []

        self.hist = {
            'rewards': [],
            'sharpe_rewards': [],
            'ep_rewards': [],
            'best_ep_rewards': [],
            'ws': []
        }

    def _generate_episode(self):
        state = self._env.reset()
        sharpe_rewards = []

        w_s = self._policy.act(self._episode_length)
        norm_ws = softmax(w_s, axis=1)
        rs = self._env.step(norm_ws, eps_len=self._episode_length, n_obs=self._obs_period)

        Rs = np.sum(rs, axis=1)

        if self._reward_mode == 'dsr':
            for R in Rs:
                self._update_dsr(R)
                sharpe_rewards.append(self._Dt)

        self.hist['rewards'].append(Rs)
        self.hist['sharpe_rewards'].append(sharpe_rewards)
        self.hist['ws'].append(w_s)
        self.hist['ep_rewards'].append(np.sum(rs))

        return np.array(Rs)

    def train(self, num_episodes=1000):
        self._policy.reset()


        for i in range(num_episodes):
            # run a single episode

            self._generate_episode()

            # update policy
            rewards = self.hist['sharpe_rewards'][i] if self._reward_mode == 'dsr' else self.hist['rewards'][i]
            self._policy.update(self.hist['ws'][i], rewards)

    def historical_test(self):
        test_data = self._env.get_test_data()
        n_samples = len(test_data)

        # calculate the weights for the policy and the baseline
        eq_ws = (1 / self._env.n_assets) * np.ones((n_samples, self._env.n_assets))
        policy_ws = self._policy.act(n_samples)
        norm_ws = softmax(policy_ws, axis=1)

        # calculate the returns from the policy and the baseline
        eq_ws__rs = np.sum(test_data * eq_ws, axis=1)
        policy_ws_rs = np.sum(test_data * norm_ws, axis=1)

        # extend the return series

        dates = list((np.array(test_data.index)[:, None]))

        self.test_dates.extend(dates)
        self.historical_eq_weight_returns.extend(eq_ws__rs)
        self.historical_policy_returns.extend(policy_ws_rs)

        self._env.slide_window()

    def test(self, t_steps=1000):
        w_s = self._policy.act(t_steps)
        rs = self._env.step(w_s, eps_len=t_steps, n_obs=1)

        return np.sum(rs, axis=1)

    def _update_dsr(self, R):
        self._At = self._Atm1 + self._η * (R - self._Atm1)
        self._Bt = self._Btm1 + self._η * (R ** 2 - self._Btm1)
        num = self._Btm1 * (R - self._Atm1) - self._Atm1 * (R ** 2 - self._Btm1) / 2
        denom = (self._Btm1 - self._Atm1 ** 2) ** (3 / 2)
        self._Dt = num / (denom + 1e-10)
        self._Atm1 = self._At
        self._Btm1 = self._Bt

    def plot_smooth_rewards(self, eps, window=10):
        rs = np.array(pd.Series(self.hist['ep_rewards']).rolling(window).mean()[window - 1:])
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(np.arange(eps - (window - 1)), rs)
        plt.ylabel('Total Returns')
        plt.xlabel('Episode')
        #         plt.plot(np.arange(eps), self.hist['best_ep_rewards'])
        plt.hlines(self._optimal_val, 0, eps - (window - 1), linestyles='dashed', colors='red')

    def plot_weights_vs_ep(self, labels=None):
        av_weights = np.array(self.hist['ws']).mean(axis=1)
        wn_over_time = list(zip(*av_weights))
        plt.figure(figsize=(8, 6), dpi=100)
        for i, weights in enumerate(wn_over_time):
            label = 'Asset {} weighting'.format(i) if not labels else labels[i]
            plt.plot(wn_over_time[i], label=label)

        plt.xlabel('Episode')
        plt.ylabel('Asset Weighting')
        plt.legend()

    def plot_episode_weights(self, ep):
        final_weights = list(zip(*np.array(self.hist['ws'])[ep - 1][3:]))
        plt.figure(figsize=(8, 6), dpi=100)
        for i, weights in enumerate(final_weights):
            plt.plot(final_weights[i], linewidth=1, label='Asset {} weighting'.format(i))

        plt.xlabel('Time Step')
        plt.ylabel('Asset Weighting');
        plt.legend()
