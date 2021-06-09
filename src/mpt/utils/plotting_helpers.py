import matplotlib.pyplot as plt


def plot_return_variance_space(R̄_π, x, y, σ_π, title, save_name):
    # plt.title(title, size=20)
    plt.plot(x, y, linewidth=1.5)
    plt.scatter(σ_π, R̄_π, marker='x', color='red', s=100, linewidth=1.5)
    plt.ylabel('$r_\pi$', size=16)
    plt.xlabel('$\sigma_\pi$', size=16)
    if save_name:
        plt.savefig(save_name)
    plt.show()

