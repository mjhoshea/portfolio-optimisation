import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import softmax


class Harness:

    def __init__(self, env, policy, episode_length=50, reward_mode='returns', η=0.05, obs_period=1):
        self._env = env
        self._policy = policy
        self._episode_length = episode_length
        self._optimal_val = max(env.μ) * episode_length

        self._obs_period=20

        # differential sharpe ratio params
        self._reward_mode = reward_mode
        self._At = 0
        self._Atm1 = 0
        self._Bt = 0
        self._Btm1 = 0
        self._Dt = 0
        self._η = η

        self.hist = {
            'rewards': [],
            'sharpe_rewards': [],
            'ep_rewards': [],
            'best_ep_rewards': [],
            'ws': [],
            'norm_ws': [],
            'a_ns': [],
            'γ': policy.γ,
            'mu': [],
            'sigma': []

        }

    def _generate_episode(self):
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
        self.hist['norm_ws'].append(norm_ws)
        self.hist['ep_rewards'].append(np.sum(rs))
        self.hist['mu'].append(self._policy.θ_μ)
        self.hist['sigma'].append(self._policy.θ_σ)

        return np.array(Rs)

    def train(self, num_episodes=1000):

        if self._reward_mode == 'dsr':
            w_s = self._policy.act(1000)
            norm_ws = softmax(w_s, axis=1)
            rs = self._env.step(norm_ws, eps_len=1000, n_obs=1)
            Rs = np.sum(rs, axis=1)
            for R in Rs:
                self._update_dsr(R)

        for i in range(num_episodes):
            # run a single episode

            if self._reward_mode == 'dsr':
                w_s = self._policy.act(100)
                norm_ws = softmax(w_s, axis=1)
                rs = self._env.step(norm_ws, eps_len=100, n_obs=1)
                Rs = np.sum(rs, axis=1)
                for R in Rs:
                    self._update_dsr(R)

            self._generate_episode()

            # update policy
            rewards = self.hist['sharpe_rewards'][i] if self._reward_mode =='dsr' else self.hist['rewards'][i]
            self._policy.update(self.hist['ws'][i], rewards)

    def test(self, t_steps=1000):
        w_s = self._policy.act(t_steps)
        norm_ws = softmax(w_s, axis=1)

        rs = self._env.step(norm_ws, eps_len=t_steps, n_obs=1)

        return np.sum(rs, axis=1)

    def _update_dsr(self, R):
        self._At = self._Atm1 + self._η * (R - self._Atm1)
        self._Bt = self._Btm1 + self._η * (R ** 2 - self._Btm1)
        num = self._Btm1 * (R - self._Atm1) - self._Atm1 * (R ** 2 - self._Btm1) / 2
        denom = (self._Btm1 - self._Atm1 ** 2) ** (3 / 2)
        self._Dt = num / denom
        self._Atm1 = self._At
        self._Btm1 = self._Bt

    def plot_smooth_rewards(self, eps, window=10):
        rs = np.array(pd.Series(self.hist['ep_rewards']).rolling(window).mean()[window - 1:])
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(np.arange(eps - (window - 1)), rs, label='Policy Return')
        plt.ylabel('Total Returns', size=12)
        plt.xlabel('Episode', size=12)
        #         plt.plot(np.arange(eps), self.hist['best_ep_rewards'])
        plt.hlines(self._optimal_val, 0, eps - (window - 1),
                   linestyles='dashed', colors='red', label='Optimal Av. Return')
        plt.legend()

    def plot_weights_vs_ep(self, labels=None):
        av_weights = np.array(self.hist['norm_ws']).mean(axis=1)
        wn_over_time = list(zip(*av_weights))
        plt.figure(figsize=(8, 6), dpi=100)
        for i, weights in enumerate(wn_over_time):
            label = 'Asset {} weighting'.format(i) if not labels else labels[i]
            plt.plot(wn_over_time[i], label=label)

        plt.xlabel('Episode', size=12)
        plt.ylabel('Asset Weighting', size=12)
        plt.legend()

    def plot_episode_weights(self, ep):
        final_weights = list(zip(*np.array(self.hist['ws'])[ep - 1][3:]))
        plt.figure(figsize=(8, 6), dpi=100)
        for i, weights in enumerate(final_weights):
            plt.plot(final_weights[i], linewidth=1, label='Asset {} weighting'.format(i))

        plt.xlabel('Time Step')
        plt.ylabel('Asset Weighting');
        plt.legend()
