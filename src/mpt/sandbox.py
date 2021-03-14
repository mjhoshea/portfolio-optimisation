from EfficientFrontier import EfficientFrontier

import numpy as np

R = np.array([14, 8, 20])

cov = np.array([
    [6 * 6, 0.5 * 6 * 3, 0.2 * 6 * 15],
    [0.5 * 6 * 3, 3 * 3, 0.4 * 3 * 15],
    [0.2 * 6 * 15, 0.4 * 3 * 15, 15 * 15]
])

R_f = 5

efficient_frontier = EfficientFrontier.from_sample_statistics(R, cov)

# efficient_frontier.plot_frontier(allow_shorts=True, allow_lending=True, riskless_rate=5)

efficient_frontier.plot_frontier(allow_shorts=False, allow_lending=False, riskless_rate=5)

