import numpy as np
from scipy import stats
from typing import Union, Iterable, Tuple
import warnings

# Type alias for the dsitribution parameters
NIGParams = Tuple[float, float, float, float]  # NIG hyperparameters: (beta1, beta2, beta3, beta4)
StudentTParams = Tuple[float, float, float]  # Student-t parameters: (df, loc, scale)


def NIGtoNormal(prior: NIGParams) -> Tuple[float, float]:
    """
    Estimates the expected Mean and Standard Deviation of the Normal distribution
    given the current NIG hyperparameters (b1, b2, b3, b4).
    """
    b1, b2, b3, b4 = prior

    # Expected Mean E[mu] = b1
    mu_hat = b1

    # Expected Variance E[sigma^2] = b4 / (b3 - 1). b3 must be > 1 for the mean of variance to exist.
    if b3 > 1:
        sigma_sq_hat = b4 / (b3 - 1)
        sigma_hat = np.sqrt(sigma_sq_hat)
    else:
        # Warn instead of crashing, return NaN for sigma
        warnings.warn(f"Beta^(3) ({b3}) <= 1, variance is undefined. Returning NaN.")
        sigma_hat = np.nan

    return mu_hat, sigma_hat


def NIGUpdate(prior: NIGParams, z: Union[float, Iterable[float]]) -> NIGParams:
    """
    Updates the NIG hyperparameters given new evidence z.
    Returns the new (posterior) parameter tuple (b1', b2', b3', b4').
    """
    b1, b2, b3, b4 = prior

    # Ensure z is an array, e.g., in case z is a single float
    z = np.atleast_1d(z).astype(float)
    n = z.size  # number of new observations

    if n == 0:
        # No new data, return prior unchanged
        return prior

    # Sample statistics
    z_bar = np.mean(z)
    S = np.sum((z - z_bar) ** 2)

    # Update equations (Appendix B)
    b2_prime = b2 + n
    b1_prime = (b2 * b1 + n * z_bar) / b2_prime
    b3_prime = b3 + (n / 2.0)
    b4_prime = b4 + (S / 2.0) + ((b2 * n) / (2.0 * b2_prime)) * ((z_bar - b1) ** 2)

    return b1_prime, b2_prime, b3_prime, b4_prime


def NIGPredictiveParameters(prior: NIGParams) -> StudentTParams:
    """
    Returns the parameters (df, loc, scale) for the Student-t
    """
    b1, b2, b3, b4 = prior

    df = 2 * b3
    loc = b1
    if b2 > 0 and b3 > 0:
        scale = np.sqrt((b4 * (b2 + 1)) / (b3 * b2))
    else:
        scale = np.nan

    return df, loc, scale


def sample_predictive(prior: NIGParams, n_samples: int = 1) -> np.ndarray:
    """
    Generates random samples from the posterior predictive Student-t distribution.
    """
    df, loc, scale = NIGPredictiveParameters(prior)

    if np.isnan(scale):
        # Return NaNs if parameters are invalid
        return np.full(n_samples, np.nan)

    return stats.t.rvs(df, loc=loc, scale=scale, size=n_samples)