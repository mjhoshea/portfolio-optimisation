import matplotlib.pyplot as plt


def plot_return_variance_space(R̄_π, x, y, σ_π, save_name, returns=None, stds=None):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(x, y, linewidth=1.5, label='Efficient Frontier')
    plt.scatter(σ_π, R̄_π, marker='x', color='red', s=100, linewidth=1.5, label='Tangency Portfolio')
    if stds:
        plt.scatter(stds, returns, marker='x', color='#f79a1e', s=100, linewidth=1.5, label='Asset')
    plt.ylabel('$r_\pi$', size=16)
    plt.xlabel('$\sigma_\pi$', size=16)
    plt.legend()
    if save_name:
        plt.savefig(save_name)
    plt.show()

