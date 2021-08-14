import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Harness:

    def __init__(self, env, policy, episode_length=50, reward_mode='returns', η=0.05):
        self._env = env
        self._policy = policy
        self._episode_length = episode_length
        self._optimal_val = max(env.μ) * episode_length

        # differential sharpe ratio params
        self._reward_mode = reward_mode
        self._At = 0
        self._Atm1 = 0
        self._Bt = 0
        self._Btm1 = 0
        self._Dt = 0
        self._η = η

        self.hist = {
            'states': [],
            'rewards': [],
            'sharpe_rewards': [],
            'ep_rewards': [],
            'best_ep_rewards': [],
            'ws': [],
            'a_ns': [],
            'γ': policy.γ,
            'α': policy.α,
            'param': policy._parameterisation,
            'grad_adpt_mode': policy._grad_adpt_mode

        }

    def _generate_episode(self):
        state = self._env.reset()

        ep_reward = 0
        best_ep_reward = 0
        ws = []
        rewards = []
        sharpe_rewards = []
        states = []

        for i in range(self._episode_length):
            X = np.eye(self._policy.n_assets)
            w, a = self._policy.act(X)

            rs = self._env.step(w)
            R = sum(rs)
            best_r = max(rs)

            if self._reward_mode == 'dsr':
                self._update_dsr(R)
                ep_reward += R
                best_ep_reward += best_r
                sharpe_rewards.append(self._Dt)
                rewards.append(R)

            else:
                ep_reward += R
                best_ep_reward += best_r
                rewards.append(R)

            states.append(X)
            ws.append(w)

        self.hist['states'].append(states)
        self.hist['rewards'].append(rewards)
        self.hist['sharpe_rewards'].append(sharpe_rewards)
        self.hist['ws'].append(ws)
        self.hist['a_ns'].append(a)
        self.hist['ep_rewards'].append(ep_reward)
        self.hist['best_ep_rewards'].append(best_ep_reward)

        return np.array(states), np.array(rewards)

    def train(self, num_episodes=1000):
        self._policy.reset()

        if self._reward_mode == 'dsr':
            for i in range(10000):
                X = np.eye(self._policy.n_assets)
                w, a = self._policy.act(X)

                R, rs, best_r = self._env.step(w)
                self._update_dsr(R)

        for i in range(num_episodes):
            # run a single episode

            if self._reward_mode == 'dsr':
                for _ in range(10):
                    X = np.eye(self._policy.n_assets)
                    w, a = self._policy.act(X)

                    R, rs, best_r = self._env.step(w)
                    self._update_dsr(R)

            self._generate_episode()

            # update policy
            rewards = self.hist['sharpe_rewards'][i] if self._reward_mode =='dsr' else self.hist['rewards'][i]
            self._policy.update(self.hist['states'][i], self.hist['ws'][i], rewards)

    def test(self, t_steps=1000):
        rewards = []
        for i in range(t_steps):
            X = np.eye(self._policy.n_assets)
            w, a = self._policy.act(X)

            R, rs, best_r = self._env.step(w)
            rewards.append(R)
        return rewards

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
