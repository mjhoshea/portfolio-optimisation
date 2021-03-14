import numpy as np


def calc_r_π(r_i, x_i, r_j, x_j):
    return r_i * x_i + r_j * x_j


def calc_σ_π(σ_i, x_i, σ_j, x_j, σ_ij):
    σ_p_sqr = σ_i ** 2 * x_i ** 2 + σ_j ** 2 * x_j ** 2 + 2 * σ_ij * x_i * x_j
    return np.sqrt(σ_p_sqr)
