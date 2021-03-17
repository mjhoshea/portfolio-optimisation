import unittest
from src.mpt.efficient_frontier.EfficientFrontier import EfficientFrontier
import numpy as np

R = np.array([14, 8, 20])
cov = np.array([
    [6 * 6, 0.5 * 6 * 3, 0.2 * 6 * 15],
    [0.5 * 6 * 3, 3 * 3, 0.4 * 3 * 15],
    [0.2 * 6 * 15, 0.4 * 3 * 15, 15 * 15]
])


class EfficientFrontierTests(unittest.TestCase):

    def test_should_calculate_correct_tangency_weights(self):
        expected_weights = [0.77777777, 0.05555555, 0.16666666]

        ef = EfficientFrontier.from_sample_statistics(R, cov)

        risk_free_rate = 5
        x = ef._tangency_portfolio_weights(risk_free_rate)

        assert np.all(np.isclose(x, expected_weights))

    def test_should_calculate_correct_tangency_weights2(self):
        expected_weights = [0.35, 0.6, 0.05]

        risk_free_rate = 2

        ef = EfficientFrontier.from_sample_statistics(R, cov)
        x = ef._tangency_portfolio_weights(risk_free_rate)

        assert np.all(np.isclose(x, expected_weights))



    def test_should_calculate_correct_return(self):

        ef = EfficientFrontier.from_sample_statistics(R, cov)
        weights = [0.35, 0.6, 0.05]
        R_p = ef._calculate_portfolio_return(weights)


        expected_portfolio_return = 10.7

        assert np.isclose(R_p,expected_portfolio_return)


    def test_should_calculate_correct_expected_volitility(self):
        expected_weights = [0.35, 0.6, 0.05]

        risk_free_rate = 2

        ef = EfficientFrontier.from_sample_statistics(R, cov)

        weights = [0.35, 0.6, 0.05]
        σ = ef._calculate_portfolio_volatility(np.array(weights))

        expected_σ = np.sqrt(5481/400)

        assert np.isclose(expected_σ, σ)

if __name__ == '__main__':
    unittest.main()
