import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

from GenerativeMarketEnv import GenerativeMarketEnv

import numpy as np
import pandas as pd

from HarnessVec import HarnessVec
from DirichletPolicyVec import DirichletPolicyVec


class ParallelRunnerVec:

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
        p = DirichletPolicyVec(*list(params.values())[:-1])
        h = HarnessVec(e, p)
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
        p = DirichletPolicyVec(*list(params.values())[:-1])
        h = HarnessVec(e, p, self._episode_length, reward_mode=self._reward_mode)
        h.train(num_episodes=params['eps'])
        rewards = h.test(100000)

        return rewards

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

            mean = ep_rewards.mean(axis=0)
            std = ep_rewards.std(axis=0)

            smooth_mean = np.array(pd.Series(mean).rolling(window).mean()[window - 1:])
            plt.plot(np.arange(ep_rewards.shape[1] - (window - 1)), smooth_mean, label=self._legend_labels[key])
            plt.fill_between(np.arange(ep_rewards.shape[1]), mean - std, mean + std, alpha=0.6)

        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Total Returns')
        plt.hlines(150, 0, ep_rewards.shape[1], linestyles='dashed', colors='red')
        if file_name:
            plt.savefig('./report_images/{}'.format(file_name))