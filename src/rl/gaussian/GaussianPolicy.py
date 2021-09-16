import numpy as np

from scipy.stats import multivariate_normal


class GaussianPolicy:

    def __init__(self, n_assets, α_μ=0.0001, α_σ=0.0001, γ=0.9):

        self.n_assets = n_assets
        self.θ_μ = np.ones(self.n_assets)
        self.θ_σ = np.ones(self.n_assets)
        self.α_μ = α_μ
        self.α_σ = α_σ

        self.ep = 0

        self.γ = γ

    def act(self, n=1):
        w_n = multivariate_normal.rvs(self.θ_μ, np.exp(self.θ_σ), n)
        return w_n

    def μ_grad(self, action):
        return (1 / np.exp(self.θ_σ) ** 2) * (action - self.θ_μ)

    def σ_grad(self, action):
        return (((action - self.θ_μ) ** 2) / (np.exp(self.θ_σ) ** 2)) - 1

    # Agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.γ + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def update(self, ws, rs):
        self.ep += 1
        μ_grads = [self.μ_grad(w) for w in ws]
        σ_grads = [self.σ_grad(w) for w in ws]

        G = self.discount_rewards(rs)

        for μ_grad, σ_grad, Gt in zip(μ_grads, σ_grads, G):
            self.θ_μ += self.α_μ * Gt * μ_grad
            self.θ_σ += self.α_μ * Gt * σ_grad

    def reset(self):
        self.θ_μ = np.ones(self.n_assets)
        self.θ_σ = np.ones(self.n_assets)
