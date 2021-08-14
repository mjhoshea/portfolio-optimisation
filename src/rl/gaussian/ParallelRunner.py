import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

from GenerativeMarketEnv import GenerativeMarketEnv

import numpy as np
import pandas as pd

from Harness import Harness
from GaussianPolicy import GaussianPolicy


class ParallelRunner:

    def __init__(self, μ, Σ, n_runs, params, episode_length, legend_labels=None, reward_mode='returns', η=0.05):
        self._μ = μ
        self._Σ = Σ
        self._n_runs = n_runs
        self._params = params
        self._legend_labels = legend_labels
        self._reward_mode = reward_mode
        self._η = η
        self._episode_length=episode_length
        self.results = defaultdict(list)

    def run(self):
        arr = []
        for i, param in tqdm(enumerate(self._params)):
            rs = Parallel(n_jobs=10, backend='loky')(delayed(self.single_run)(param) for i in tqdm(range(self._n_runs)))
            arr.append(rs)
        for i, r in enumerate(arr):
            self.results[str(i)] = r

    def single_run(self, params):
        e = GenerativeMarketEnv(self._μ, self._Σ)
        p = GaussianPolicy(*list(params.values())[:-1])
        h = Harness(e, p)
        h.train(num_episodes=params['eps'])

        return h.hist

    def run_test(self):
        arr = []
        for i, param in tqdm(enumerate(self._params)):
            rs = Parallel(n_jobs=10, backend='loky')(delayed(self.single_run_test)(param) for i in tqdm(range(self._n_runs)))
            arr.append(rs)
        for i, r in enumerate(arr):
            self.results[str(i)] = r

    def single_run_test(self, params):
        e = GenerativeMarketEnv(self._μ, self._Σ)
        p = GaussianPolicy(*list(params.values())[:-1])
        h = Harness(e, p, self._episode_length, reward_mode=self._reward_mode)
        h.train(num_episodes=params['eps'])
        rewards = h.test(100000)

        return rewards


    def plot_av_weights(self, window=1, file_name=None):

        plt.figure(figsize=(8, 6), dpi=100)

        for key in self.results:
            norm_weights = []
            for r in self.results[key]:
                norm_weights.append(r['norm_ws'])
            norm_weights = np.array(norm_weights)

            mean = norm_weights.mean(axis=(0,2))
            std = norm_weights.std(axis=(0,2))

            for i in range(3):
                plt.plot(np.arange(1000), mean[:, i], label='Asset {}'.format(i+1))
                lb = mean[:, i] - std[:, i]
                ub = mean[:, i] + std[:, i]
                lb[lb < 0] = 0
                ub[ub > 1] = 1

                plt.fill_between(np.arange(1000), lb, ub, alpha=0.6)

            # plt.plot(np.arange(1000), mean[:, 1], label='Asset 2 Weight')
            # plt.fill_between(np.arange(1000), mean[:, 1] - std[:, 1], mean[:, 1] + std[:, 1], alpha=0.6)
            #
            # plt.plot(np.arange(1000), mean[:, 2], label='Asset 1 Weight')
            # plt.fill_between(np.arange(1000), mean[:, 2] - std[:, 2], mean[:, 2] + std[:, 2], alpha=0.6)

        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Average Asset Weightings')
        if file_name:
            plt.savefig('./report_images/{}'.format(file_name))

    def plot_av_reward_for_run(self, key):

        ep_rewards = []
        plt.figure(figsize=(8, 6), dpi=100)
        for r in self.results[key]:
            ep_rewards.append(r['ep_rewards'])
        ep_rewards = np.array(ep_rewards)

        plt.figure(figsize=(8, 6), dpi=100)
        mean = ep_rewards.mean(axis=0)
        std = ep_rewards.std(axis=0)

        plt.plot(np.arange(ep_rewards.shape[1]), mean)
        plt.hlines(150, 0, ep_rewards.shape[1], linestyles='dashed', colors='red')
        plt.fill_between(np.arange(ep_rewards.shape[1]), mean - std, mean + std, alpha=0.6)


    def plot_av_rewards(self, window=1, file_name=None):
        plt.figure(figsize=(8, 6), dpi=100)
        for key in self.results:
            ep_rewards = []
            for r in self.results[key]:
                ep_rewards.append(r['ep_rewards'])
            ep_rewards = np.array(ep_rewards)
            print(ep_rewards.shape)

            mean = ep_rewards.mean(axis=0)
            std = ep_rewards.std(axis=0)

            smooth_mean = np.array(pd.Series(mean).rolling(window).mean()[window - 1:])

            plt.plot(np.arange(ep_rewards.shape[1] - (window - 1)), smooth_mean)
            plt.fill_between(np.arange(ep_rewards.shape[1]), mean - 2*std, mean + 2*std, alpha=0.6)

        plt.xlabel('Episode')
        plt.ylabel('Average Episodic Returns')
        plt.hlines(150, 0, ep_rewards.shape[1], linestyles='dashed', colors='red')
        if file_name:
            plt.savefig('./report_images/{}'.format(file_name))