"""
Implementation of the execution time metric for the grey-box models.

Author: Juan-Pablo Futalef

"""
import dill as pickle
import numpy as np
from pathlib import Path
import tqdm
import multiprocessing as mp


def computational_load(t_sim, t_exec):
    """
    Fits the passed time array to a line y = mx and returns the slope m
    :param t_sim: the simulation time array
    :param t_exec: the execution time array
    :return: the slope m
    """
    # Fit the simulation time array to a line
    t_sim_col = t_sim[:, np.newaxis]
    m, _, _, _ = np.linalg.lstsq(t_sim_col, t_exec, rcond=None)
    return m[0]


def get_execution_time_data(sim_data):
    """
    Get the execution time array in seconds
    :param sim_data: dictionary of simulation data
    :return: the array starting at zero
    """
    # The execution time array
    t_exec = np.array(sim_data["execution_time_array"])
    t_exec = t_exec - t_exec[0]

    # The simulation time array
    t_sim = np.array(sim_data["time"])

    # Make both arrays the same size by removing the last elements
    if len(t_sim) > len(t_exec):
        t_sim = t_sim[:len(t_exec)]
    elif len(t_exec) > len(t_sim):
        t_exec = t_exec[:len(t_sim)]

    return t_sim, t_exec


def computational_load_folder(folder):
    """
    Iterates all files in the folder and computes the average computational load
    :return: the average computational load
    """
    # Get all files in the folder
    files = [x for x in Path(folder).iterdir() if x.is_file() and x.suffix == ".pkl"]

    # Get cores
    n_cores = mp.cpu_count()

    # Do task with tqdm and pool
    with mp.Pool(n_cores) as pool:
        results = list(tqdm.tqdm(pool.imap(read_computational_load, files), total=len(files)))

    # Iterate the results
    cum_comp_load = 0
    n_files = 0
    info = {}
    for file, result in tqdm.tqdm(zip(files, results), total=len(files)):
        if result is None:
            continue

        # Save info
        info[file] = {"comp_load": result[0],
                      "sim_time": result[1],
                      "exec_time": result[2]}

        cum_comp_load += result[0]
        n_files += 1

    # Compute
    avg_comp_load = cum_comp_load / n_files if n_files > 0 else 0

    print(f"    Processed {n_files} files")
    print(f"    Total cumulative computational load: {cum_comp_load}")
    print(f"    Average computational load: {avg_comp_load}")

    return cum_comp_load / n_files, info


def read_computational_load(file_path):
    try:
        with open(file_path, "rb") as f:
            sim_data = pickle.load(f)

        t_sim, t_exec = get_execution_time_data(sim_data)
        comp_load = computational_load(t_sim, t_exec)
        return comp_load, t_sim, t_exec

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def folders_comparison(folders,
                       names=None,
                       save_to=None):
    """
    Compares the computational load between folders
    :param folders: list of folders
    :return: the comparison
    """
    import pandas as pd
    info = {}
    av_loads = {}
    for folder in folders:
        print(f"------- Computing for {folder} -------")
        avg_load, folder_info = computational_load_folder(folder)
        folder_info["avg_load"] = avg_load
        info[folder] = folder_info

        # Save the results to the table
        av_loads[folder] = avg_load

    # create table
    df = pd.DataFrame.from_dict(av_loads, orient="index")
    df.columns = pd.Index(["L1"])
    if names is not None:
        df.index = pd.Index(names, name="Model")

    if save_to is not None:
        df.to_csv(save_to / "computational_load.csv")

        with open(save_to / "computational_load_info.pkl", "wb") as f:
            pickle.dump(info, f)

    return df, info
