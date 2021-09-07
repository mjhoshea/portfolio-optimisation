from scipy.stats import dirichlet
from scipy.special import digamma, polygamma

import numpy as np


class DirichletPolicyVec:

    def __init__(
            self, n_assets, α=0.001, α_end=0.0001, start_ep=1000,
            α_decay_steps=1000, γ=0.9, grad_adpt_mode='poly',
            returns_adpt_mode='both', parameterisation='linear'
    ):
        self.n_assets = n_assets
        self.θ = np.ones(self.n_assets)
        self.α = α
        self.α_start = α
        self.α_end = α_end

        self.ep = 0
        self.start_ep = start_ep
        self.α_decay_steps = α_decay_steps

        self.γ = γ
        self.a_min = 1e-2
        self.a_max = 20
        self._grad_adpt_mode = grad_adpt_mode
        self._returns_adpt_mode = returns_adpt_mode
        self._parameterisation = parameterisation

    def act(self, a_n, n=1):
        w_n = dirichlet.rvs(a_n, n)
        return w_n

    def grad(self, w):
        w[w <= 0] = 1e-10
        a_n = self.calc_an()
        g = (digamma(np.sum(a_n)) - digamma(a_n) + np.log(w)).T

        if self._parameterisation == 'linear':
            g = g
        elif self._parameterisation == 'exp':
            g = g * np.exp(a_n)
        elif self._parameterisation == 'softplus':
            g = g * (1 / (1 + np.exp(a_n)))

        apt_g = self._adapt_grad(g, a_n)
        return apt_g

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.γ + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def update(self, ws, rs):
        self.ep += 1
        self.α = self._decay_epsilon()
        grads = [self.grad(w) for w in ws]
        G = self.discount_rewards(rs)
        adpt_G = self._adapt_returns(rs, G)
        for grad, Gt in zip(grads, adpt_G):
            self.θ += self.α * (Gt) * grad

    def reset(self):
        self.θ = np.ones(self.n_assets)

    def _decay_epsilon(self):
        alpha = self.α_end + (self.α_start - self.α_end) * \
                np.exp(-1. * ((self.ep - self.start_ep) / self.α_decay_steps))
        return min(alpha, self.α_start)

    def calc_an(self):
        if self._parameterisation == 'linear':
            a_n = self.θ
            a_n[a_n < self.a_min] = self.a_min
            a_n[a_n > self.a_max] = self.a_max
            return a_n
        elif self._parameterisation == 'exp':
            return np.exp(self.θ)
        elif self._parameterisation == 'softplus':
            return np.log((1 + 1e-10) + np.exp(self.θ))
        else:
            raise ValueError('Invalid value for parameterisation.')

    def _adapt_grad(self, g, a_n):
        scale = 1
        if self._grad_adpt_mode == 'max_digamma':
            scale = max(abs(digamma(a_n)))
        elif self._grad_adpt_mode == 'max_polygamma':
            scale = max(abs(polygamma(1, a_n)))
        elif self._grad_adpt_mode == 'max_di_polly_interpolation':
            scale_α = max(abs(digamma(a_n)))
            scale_β = max(abs(polygamma(1, a_n)))
            scale = .5 * scale_α + .5 * scale_β
        elif self._grad_adpt_mode == 'natural_gradient':
            F = np.diag(polygamma(1, a_n)) - polygamma(1, np.sum(a_n)) * np.ones((self.n_assets, self.n_assets))
            F_inv = np.linalg.inv(F)
            return F_inv @ g
        elif not self._grad_adpt_mode:
            return g
        else:
            raise ValueError('Invalid value for returns_adpt_mode.')

        return g / scale

    def _adapt_returns(self, rs, G):
        b = np.mean(rs)
        σ = np.var(rs)
        if self._returns_adpt_mode == 'avg_r_baseline':
            return G - b
        elif self._returns_adpt_mode == 'noise_whitening':
            return G / (σ + 10e-10)
        elif self._returns_adpt_mode == 'both':
            return (G - b) / (σ + 10e-10)
        else:
            return G








