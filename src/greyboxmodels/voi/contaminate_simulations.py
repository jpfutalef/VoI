"""
A module that can read datasets_development of simulations and output new ones with contaminated data.

Author: Juan-Pablo Futalef
"""
import numpy as np
import pandas as pd
import dill as pickle
from import import Path

def folder_state_data(folder: Path):
    from gre