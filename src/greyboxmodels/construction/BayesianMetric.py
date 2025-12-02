"""
Set of tools to do Value of Information tasks.

Author: Juan-Pablo Futalef
"""

import numpy as np
from scipy.stats import norm
from typing import Iterable, Union


class Metric:
    def __init__(self, value=None):
        """
        Initializes the Metric class.

        :param value: A simple numeric value.
        """
        self.value = value

    def get(self):
        """
        Returns the metric value.

        :return: The metric value.
        """
        return self.value

    def update(self, value):
        """
        Updates the metric value.

        :param value: The new value to update.
        """
        self.value = value


class BayesianNormalEstimator(Metric):
    def __init__(self,
                 mu0: float,
                 n0: float,
                 alpha0: float,
                 beta0: float):
        super().__init__()
        # Initialize lecture-style hyperparameters
        self.mu0 = mu0
        self.n0 = n0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Track histories
        self.mu_history = [mu0]
        self.sigma_history = [self._current_sigma()]

    def _current_sigma(self):
        # E[sigma^2] = beta0/(alpha0 - 1) for alpha0>1
        if self.alpha0 > 1:
            return np.sqrt(self.beta0 / (self.alpha0 - 1))
        return np.nan

    def update(self, z):
        """
        Batch Bayesian update for n observations

        :param z: 1D array of observations of length n.
        """
        z = np.asarray(z, dtype=float)
        n = z.size
        if n == 0:
            return  # nothing to update

        # Sample mean and sum of squares
        x_bar = z.mean()
        S = ((z - x_bar) ** 2).sum()

        # Batch update
        n_n = self.n0 + n
        mu_n = (self.n0 * self.mu0 + n * x_bar) / n_n
        alpha_n = self.alpha0 + n / 2
        beta_n = (self.beta0 + 0.5 * S + (self.n0 * n) / (2 * n_n) * (x_bar - self.mu0) ** 2)

        # Assign posterior to new prior
        self.mu0 = mu_n
        self.n0 = n_n
        self.alpha0 = alpha_n
        self.beta0 = beta_n

        # Record history
        self.mu_history.append(self.mu0)
        self.sigma_history.append(self._current_sigma())

    def get(self):
        """Return current posterior mean mu_n."""
        return self.mu0

    def get_mean_variance(self):
        """Return (mu_n, sigma_n)."""
        return self.mu0, self._current_sigma()

    def get_history(self):
        """Return arrays of mu and sigma history."""
        return np.array(self.mu_history), np.array(self.sigma_history)
