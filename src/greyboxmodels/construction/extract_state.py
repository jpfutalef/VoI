"""
A module that can read datasets_development of simulations and output new ones with contaminated data.

Author: Juan-Pablo Futalef
"""
import numpy as np
import pandas as pd
import dill as pickle
from pathlib import Path
import multiprocessing as mp


def folder_state_data(folder: Path):
    """
    Get the state data from a folder
    :param folder: the folder to get the state data from
    :return: the state data
    """
    # Containers
    T = []
    X = []

    # Get the files
    files = [file for file in folder.iterdir() if file.suffix == ".pkl"]

    # Number of cores
    n_cores = mp.cpu_count()

    # Create a pool of workers
    pool = mp.Pool(n_cores)

    # Parallelize the process
    results = pool.map(get_state_array, files)

    # Close the pool
    pool.close()

    # Iterate the results to get the data
    for r in results:
        if r is None:
            continue
        T.append(r[0])
        X.append(r[1])

    return T, X


def get_state_array(filepath: Path):
    """
    Get the state array from a file
    :param filepath: the path to the file
    :return: the time array, and state matrix
    """
    try:
        # Load the data
        with open(filepath, "rb") as f:
            sim_data = pickle.load(f)

        # Get the time array
        t = np.array(sim_data["time"])

        # Get the state matrix
        state = np.array(sim_data["state"])

        return t, state

    except Exception as e:
        return None

