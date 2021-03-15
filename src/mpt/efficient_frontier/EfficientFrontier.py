import numpy as np
import random
import cvxopt as opt
import matplotlib.pyplot as plt

from plotting_helpers import plot_return_variance_space
from common_functions import calc_r_π, calc_σ_π

opt.solvers.options['show_progress'] = False


class EfficientFrontier:

    def __init__(self, μ, Σ):
        self.μ = μ
        self.Σ = Σ
        self.min_μ = min(μ)
        self.max_μ = max(μ)
        self.min_σ = min(np.diagonal(Σ))
        self.max_σ = max(np.diagonal(Σ))

    @staticmethod
    def from_sample_statistics(avg_returns, covariances):
        return EfficientFrontier(avg_returns, covariances)

    @staticmethod
    def from_raw_returns(returns):
        pass

    def plot_frontier(self, allow_shorts=True, allow_lending=True, riskless_rate=5):

        if allow_shorts and allow_lending:
            self._plot_unconstrained_frontier(riskless_rate)

        elif allow_shorts and not allow_lending:
            self._plot_no_lending_frontier()

        elif not allow_shorts and allow_lending:
            self._plot_no_shorts_frontier(riskless_rate)

        else:
            self._plot_fully_constrained_frontier()

    def _plot_unconstrained_frontier(self, riskless_rate):

        X = self._tangency_portfolio_weights(riskless_rate)

        R̄_π = self._calculate_portfolio_return(X)
        σ_π = self._calculate_portfolio_volatility(X)

        θ = (R̄_π - riskless_rate) / σ_π

        x = np.arange(0, 2 * σ_π, 0.01)
        y = [riskless_rate + θ * i for i in x]

        title = 'Efficient Frontier: Short Selling and Riskless \n Lending/Borrowing Allowed'
        plot_return_variance_space(R̄_π, x, y, σ_π, title)

    def _plot_no_lending_frontier(self):

        r_a, r_b = (5, 2)

        # calculate the tangency portfolio at first rate
        X_a = self._tangency_portfolio_weights(r_a)
        R̄_a = self._calculate_portfolio_return(X_a)
        σ_a = self._calculate_portfolio_volatility(X_a)

        # calculate the tangency portfolio at the second rate
        X_b = self._tangency_portfolio_weights(r_b)
        R̄_b = self._calculate_portfolio_return(X_b)
        σ_b = self._calculate_portfolio_volatility(X_b)

        # calculate expected return and std on combined portfolio
        X_mix = 0.5 * X_a + 0.5 * X_b
        σ_mix = self._calculate_portfolio_volatility(X_mix)
        # print(σ_mix)

        # calculate the covariance between portfolio a and b
        num = σ_mix ** 2 - (0.5 ** 2) * (σ_a ** 2) - (0.5 ** 2) * (σ_b ** 2)
        den = 2 * 0.5 * 0.5
        σ_ab = num / den

        r_π = []
        σ_π = []

        for x_a in np.arange(-2, 1, 0.1):
            x_b = 1 - x_a
            r_π.append(calc_r_π(R̄_a, x_a, R̄_b, x_b))
            σ_π.append(calc_σ_π(σ_a, x_a, σ_b, x_b, σ_ab))

        title = 'Efficient Frontier: Short Selling and Riskless \n Lending/Borrowing Not Allowed'

        plt.title(title)
        plt.plot(σ_π, r_π)
        #   plt.scatter(σ_π, R̄_π, marker='x', color='red', s=100)
        plt.ylabel('$r_p$', size=14)
        plt.xlabel('$\sigma_p$', size=14)

        plt.show()

    def _plot_no_shorts_frontier(self, riskless_rate):

        n_s = len(self.μ)
        A = opt.matrix(np.transpose(np.array(self.μ) - riskless_rate)[None, :])
        b = opt.matrix(np.array([1.]))
        P = opt.matrix(self.Σ)
        q = opt.matrix(np.zeros((n_s, 1)))
        G = opt.matrix(-np.identity(n_s))
        h = opt.matrix(np.zeros((n_s, 1)))
        sol = opt.solvers.qp(P, q, G, h, A, b)

        X = np.array(sol['x'])
        X = X/sum(X)

        R̄_π = self._calculate_portfolio_return(X)
        σ_π = self._calculate_portfolio_volatility(X)

        θ = (R̄_π - riskless_rate) / σ_π

        x = np.arange(0, 2 * σ_π, 0.01)
        y = [riskless_rate + θ[0] * i for i in x]

        title = 'Efficient Frontier: Riskless Lending and Borrowing \n ' \
                'With No Short Sales Allowed'
        plot_return_variance_space(R̄_π, x, y, σ_π, title)

        pass

    def _plot_fully_constrained_frontier(self):
        n = len(self.μ)

        # minimize
        P = opt.matrix(self.Σ)
        q = opt.matrix(np.zeros((n, 1)))

        # subject to inequality constraints
        G = opt.matrix(-np.identity(n))
        h = opt.matrix(np.zeros((n, 1)))

        # subject to equality constraints
        A = opt.matrix(np.concatenate((
            np.transpose(np.array(self.μ))[None, :],
            np.transpose(np.ones((n, 1)))), 0))

        min_μ = min(self.μ)
        max_μ = max(self.μ)
        R_π = []
        σ_π = []
        for r in np.arange(min_μ, max_μ, 0.001):
            b = opt.matrix(np.array([r, 1]))
            sol = opt.solvers.qp(P, q, G, h, A, b)
            weights = np.array(sol['x'])
            R_π.append(weights.T @ self.μ)
            σ_π.append((weights.T @ self.Σ @ weights)[0])

        σs = np.diag(self.Σ)

        title = 'Efficient Frontier: No Short Selling or Riskless \n Lending/Borrowing Not Allowed'

        plt.title(title, size=18)
        plt.plot(np.sqrt(σ_π), R_π)
        plt.scatter(np.sqrt(σs), self.μ, marker='x', color='red', s=100)

        plt.ylabel('$r_p$', size=14)
        plt.xlabel('$\sigma_p$', size=14)
        plt.show()

    def _calculate_portfolio_volatility(self, X_b):
        σ_b = np.sqrt(X_b.T @ self.Σ @ X_b)
        return σ_b

    def _calculate_portfolio_return(self, X_b):
        R̄_b = self.μ @ X_b
        return R̄_b

    def _tangency_portfolio_weights(self, riskless_rate):
        """ Tangency portfolio weight calculation.

        Calculates the weights of the tangency portfolio given a risk free rate available
        to an investor.

        This is done by absorbing the constraint of an investor being fully invested into the objective
        function being solved:

            θ = x^Tμ - R_f / x^tΣx

        And then solving the resulting system of linear equations before normalizing the weights.

        :param riskless_rate:
        :return: tangency portfolio weights
        """
        n_s = len(self.μ)
        B = self.μ - riskless_rate
        A = self.Σ * np.eye(n_s) + (np.ones(n_s) - np.eye(n_s)) * self.Σ
        Z = np.linalg.inv(A).dot(B)
        X = Z / sum(Z)
        return X

    def _plot_no_short_frontier(self):
        pass

    def _plot_constrained_frontier(self):
        pass
