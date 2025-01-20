import numpy as np
from pathlib import Path
import dill as pickle
import os
import pandas as pd
import tqdm
from typing import List
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt
import json


def ks_statistic(data1, data_ref, n_bins=50):
    """
    Compute the Kolmogorov-Smirnov statistic between two empirical cumulative distribution functions.

    """
    # Generate empirical PDF
    bins = np.linspace(min(data1.min(), data_ref.min()), max(data1.max(), data_ref.max()), n_bins)
    epdf1, _ = np.histogram(data1, bins=bins, density=True)
    epdf_ref, _ = np.histogram(data_ref, bins=bins, density=True)

    # Compute the empirical CDF
    ecdf_1 = empirical_cdf(bins, epdf1)
    ecdf_2 = empirical_cdf(bins, epdf_ref)

    # Compute the absolute difference
    abs_dff = np.abs(ecdf_1 - ecdf_2)

    # get max and the value where it happens
    ks_value = np.max(abs_dff)
    max_diff_idx = np.argmax(abs_dff)
    location = ecdf_1[max_diff_idx]

    # Store information
    info = {"location": location,
            "ks_value": ks_value,
            "bins": bins,
            "epdf1": epdf1,
            "epdf_ref": epdf_ref,
            "ecdf1": ecdf_1,
            "ecdf_ref": ecdf_2,
            "abs_diff": abs_dff,
            }

    return ks_value, info


def empirical_cdf(bins, epdf):
    """
    Compute the empirical cumulative distribution function of a dataset.
    """
    # Empirical CDF
    ecdf = np.cumsum(epdf) * np.diff(bins)

    # Add zero and one to teh array to ensure we obtain an ECDF
    ecdf = np.concatenate(([0], ecdf, [1]))

    return ecdf


def lack_of_fit(data,
                reference_data,
                state_filter: callable = None,
                ):
    """
    Compute the lack of fit between two datasets_development.
    :param data: dictionary of data points indexed by time
    :param reference_data: dictionary of reference data points indexed by time
    :param state_filter: a function to filter the states to compare
    :return: the lack of fit
    """
    # Loop over the values to compute the empirical distributions
    ks_data = {}
    for i_X, dict_Xi in reference_data.items():
        # Check if the state should be filtered
        if state_filter is not None:
            if state_filter(i_X):
                continue

        # Check if the key exists; otherwise, create it
        if i_X not in ks_data:
            ks_data[i_X] = []

        # Iterate over the time
        for tk, ref_data_array in dict_Xi.items():
            # try getting the values
            try:
                data_array = data[i_X][tk]

                # Compute the KS statistic
                ks, info = ks_statistic(data_array, ref_data_array)

                # Append the values
                ks_data[i_X].append([tk, ks, info, "no issues"])

            except Exception as e:
                ks_data[i_X].append([tk, np.nan, None, str(e)])

    # Compute the average KS for each state in time
    avg_ks_per_state = {}
    for i_X, data in ks_data.items():
        avg_ks_per_state[i_X] = np.nanmean([ks for _, ks, _, _ in data])

    # The total average KS
    avg_ks = np.nanmean([ks for ks in avg_ks_per_state.values()])

    return avg_ks, avg_ks_per_state, ks_data


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

    # # Get the files
    # files = [file for file in folder.iterdir() if file.suffix == ".pkl"]
    #
    # # Iterate over the files in the folder and get the state data
    # for file in tqdm.tqdm(files):
    #     # Get the state array
    #     try:
    #         t, state = get_state_array(file)
    #
    #     except Exception as e:
    #         print(f"Error reading {file}:\n  {e}")
    #         continue
    #
    #     # Append the data
    #     T.append(t)
    #     X.append(state)

    return T, X


def empirical_probability_density_distributions(t: List[np.ndarray],
                                                x: List[np.ndarray],
                                                n_bins: int = 50,
                                                filter: callable = None,
                                                ):
    """
    Compute the empirical distributions of a dataset. Each element in the t and x lists should come from a different
    simulation. The function returns a dictionary with the empirical distributions indexed by time.
    :param t: list of time arrays
    :param x: list of state matrices
    :param n_bins: the number of bins to use
    :return: a dictionary with the empirical distributions indexed by time. The keys are the state indices and the
    the values are lists of empirical distributions
    """
    # Create a dictionary to store the raw values
    values = {}
    issues = {}
    for i_sim, (t_sim, x_sim) in tqdm.tqdm(enumerate(zip(t, x)), total=len(t)):
        # Iterate over the time
        for k, tk in enumerate(t_sim):
            # Get the state at time tk
            X = x_sim[k, :]
            # Iterate over the state
            for i_X, Xi_tk in enumerate(X):
                # Check if the keys exist; otherwise, create them
                if i_X not in values:
                    values[i_X] = {}

                if tk not in values[i_X]:
                    values[i_X][tk] = []

                # Check for issues
                if np.isnan(Xi_tk):
                    issues[(i_sim, k, i_X)] = f"NaN value at time {tk} of state {i_X}"
                    continue

                if np.isinf(Xi_tk):
                    issues[(i_sim, k, i_X)] = f"Inf value at time {tk} of state {i_X}"
                    continue

                if Xi_tk is None:
                    issues[(i_sim, k, i_X)] = f"None value at time {tk} of state {i_X}"
                    continue

                # Append the value
                values[i_X][tk].append(Xi_tk)

    # Sort the keys of the first level
    values = dict(sorted(values.items()))

    # Sort the keys of the second level and turn the lists into arrays
    for key in values:
        values[key] = dict(sorted(values[key].items()))
        for tk in values[key]:
            values[key][tk] = np.array(values[key][tk])

    # Loop over the values to compute the empirical distributions
    epdfs = {}
    for i_X, dict_Xi in values.items():
        if filter is not None:
            if filter(i_X):
                continue
        epdfs[i_X] = {}
        for tk, array_Xi_tk in dict_Xi.items():
            # Use the list to compute the empirical distribution
            epdf, bins = np.histogram(array_Xi_tk, bins=n_bins, density=True)

            # Store the values
            epdfs[i_X][tk] = (bins, epdf)

    return epdfs, values, issues


def read_values_data(values_filepath: Path,
                     state_data_filepath: Path = None,
                     origin_folder: Path = None,
                     ):
    """
    Attempts to open the distribution data at the specified location.
    If not found, use the state data at the specified location to compute the distributions.
    As a last resort, use the reference folder to compute the distributions.
    If nothing works, raise an exception.
    :param values_filepath: the path to the values data
    :param state_data_filepath: the path to the state data (optional)
    :param origin_folder: the folder to compute the distributions from (optional)
    :return: the distributions
    """
    # Setup locations
    values_filepath = Path(values_filepath)
    save_in = values_filepath.parent

    state_data_filepath = Path(state_data_filepath) if state_data_filepath is not None else None
    origin_folder = Path(origin_folder) if origin_folder is not None else None
    distributions_filepath = save_in / f"{values_filepath.stem}_distributions.pkl"
    distributions_issues_filepath = save_in / f"{values_filepath.stem}_issues.pkl"

    # Try opening; otherwise, compute
    try:
        print(f" Attempting to open data points from: {values_filepath}")
        with open(values_filepath, "rb") as f:
            values = pickle.load(f)

    except FileNotFoundError:
        # Try opening the time and state data to compute the distributions
        try:
            print(f"    Not found... Attempting to open state data...")
            with open(state_data_filepath, "rb") as f:
                state_data = pickle.load(f)
                T = state_data["time"]
                X = state_data["state"]

        except FileNotFoundError:
            # Get the state data from the folder
            print(f"    Not found... Attempting to retrieve state data from folder...")
            T, X = folder_state_data(origin_folder)
            state_data = {"time": copy.deepcopy(T), "state": copy.deepcopy(X)}

            # Save
            with open(state_data_filepath, "wb") as f:
                pickle.dump(state_data, f)

        print(f"    Done!   Computing values from state data...")
        distributions, values, issues = empirical_probability_density_distributions(T, X)

        # Save the values
        with open(values_filepath, "wb") as f:
            pickle.dump(values, f)

        # Save the distributions
        with open(distributions_filepath, "wb") as f:
            pickle.dump(distributions, f)

        # Save the issues
        with open(distributions_issues_filepath, "wb") as f:
            pickle.dump(issues, f)

    except Exception as e:
        print(f"Error reading distributions from {values_filepath}:\n  {e}")
        raise e

    print("    Done!")

    return values


def folders_comparison(folders,
                       reference_folder,
                       save_in,
                       names=None,
                       state_filter=None,
                       ):
    """
    Compares the lack of fit between folders
    :param folders: the folders to compare
    :param reference_folder: the reference folder
    :param save_in: the folder to save the results
    :param names: the names of the models in the folders
    :param state_filter: a function to filter the states to compare
    :return: the comparison
    """
    # Get the distributions
    values_filepath = save_in / "reference_values.pkl"
    state_data_filepath = save_in / "reference_state_data.pkl"

    reference_values = read_values_data(values_filepath,
                                        state_data_filepath=state_data_filepath,
                                        origin_folder=reference_folder)

    # Compute the lack of fit for the rest of the folders using the reference distributions
    data = {}
    info = {}
    for i, folder in enumerate(folders):
        print(f"    ------- Computing for {folder} -------")
        # Specify the location to save the distributions
        model_name = names[i] if names is not None else f"model_{i}"
        values_filename = f"{model_name}_values"
        target_values_filepath = save_in / f"{values_filename}.pkl"
        state_data_filepath = save_in / f"{model_name}_state_data.pkl"

        # Get the distributions
        target_values = read_values_data(target_values_filepath,
                                         state_data_filepath=state_data_filepath,
                                         origin_folder=folder)

        # Compute the lack of fit
        print(f"    Computing the lack of fit...")
        val, val_per_state, lof_info = lack_of_fit(target_values,
                                                   reference_values,
                                                   state_filter=state_filter)

        # Save the results to the table
        data[folder] = val
        info[folder] = {"per state": val_per_state, "detailed_info": lof_info}

        print(f"    Done!")

    # create table
    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = ["Lack of fit"]
    if names is not None:
        df.index = pd.Index(names, name="model")

    # Save the table
    table_filepath = save_in / "lack_of_fit.csv"
    df.to_csv(table_filepath)

    # Save the info
    info_filepath = save_in / "lack_of_fit_info.pkl"
    with open(info_filepath, "wb") as f:
        pickle.dump(info, f)

    return df, info
