"""
A module for calculating risk metrics and performing risk analysis.

Author: Juan-Pablo Futalef
"""
import numpy as np
from scipy.integrate import simpson

def aggregate_integral(t: np.ndarray, r: np.ndarray) -> float:
    """
    Aggregates the risk metric r using integration.
    :param t: time points
    :param r: risk metric values
    :return: aggregated risk metric
    """
    return simpson(r, t)