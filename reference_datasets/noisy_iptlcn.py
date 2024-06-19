"""
Injects additive noise into the continuous variables of the IPTLCN simulations created using the WBM

Author: Juan-Pablo Futalef
"""

# %% Import libraries
import numpy as np
import pandas as pd
from pathlib import Path

# %% Specify paths
origin_dir = Path("data/simulation_references/iptlc/MonteCarlo/2024-05-09_15-16-30")
destination_dir = origin_dir / "noisy"

# %% Load the reference data

