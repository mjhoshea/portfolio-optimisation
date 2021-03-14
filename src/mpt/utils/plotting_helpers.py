import matplotlib.pyplot as plt


def plot_return_variance_space(R̄_π, x, y, σ_π, title):
    plt.title(title)
    plt.plot(x, y)
    plt.scatter(σ_π, R̄_π, marker='x', color='red', s=100)
    plt.ylabel('$r_\pi$', size=14)
    plt.xlabel('$\sigma_\pi$', size=14)
    plt.show()
