"""
A class that represents a dataset of ground truth data and implements several methods to access and manipulate it.
The dataset is supposed to exclusively be used for the construction of grey-box models.

Author: Juan-Pablo Futalef
"""
import copy
from pathlib import Path
from typing import List
import dill as pickle
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import tqdm

from greyboxmodels.modelbuild import Input
from greyboxmodels.simulation import Simulator


def load_process_simulation_file(path):
    s = Simulator.load_simulation_data(path)
    try:
        s = process_scenario(s)
    except Exception as e:
        print(f"Error processing scenario: {path}")
        print(e)
        return None
    return s

def process_scenario(scenario):
    """
    Extracts only relevant data from the scenario and processes it.
    :param scenario: A dictionary containing the scenario data.
    """
    # Extract relevant data
    processed_scenario = copy.deepcopy(scenario)
    del processed_scenario["step_data"]

    # Convert specified to numpy arrays
    keys = ['time', 'uncontrolled_inputs', 'state', 'execution_time_array']
    for key in keys:
        processed_scenario[key] = np.array(processed_scenario[key])

    return processed_scenario


def process_scenarios(scenario_list):
    """
    Processes the scenarios in the list
    :param scenario_list: A list of dictionaries, each containing an scenario of simulation data.
    :return: A list of dictionaries, each containing the processed data for the simulation
    """
    processed_scenarios = []
    for scenario in tqdm.tqdm(scenario_list):
        processed_scenario = process_scenario(scenario)
        processed_scenarios.append(processed_scenario)

    return processed_scenarios


class GroundTruthDataset:
    def __init__(self,
                 scenario_list: List[dict],
                 process_data: bool = False
                 ):
        """
            Initializes the dataset.

            :param scenario_list: A list of dictionaries, each containing an scenario of simulation data.
            :param store_raw_data: If True, the original data will be stored in the object.
            """
        # Process the scenarios
        self.scenarios = process_scenarios(scenario_list) if process_data else scenario_list

    @classmethod
    def load(cls, path):
        """
        Loads a GroundTruthDataset object from a file.

        :param path: Path to the file
        :return: A GroundTruthDataset object
        """
        with open(path, "rb") as file:
            instance = pickle.load(file)

        return instance

    @classmethod
    def load_list(cls,
                  path_list,
                  process=True,
                  parallel=True,
                  ):
        """
        Loads a list of datasets from a list of files.

        :param path_list: List of paths to the files
        :return: A list of GroundTruthDataset objects
        """
        worker = load_process_simulation_file if process else Simulator.load_simulation_data
        if not parallel:
            return cls([worker(path) for path in path_list], False)

        # Use multiprocessing to load datasets in parallel
        n_processes = min(mp.cpu_count(), len(path_list))  # Don't create more processes than needed

        def update_progress(_):
            """Callback function to update tqdm progress bar."""
            pbar.update(1)

        with mp.Pool(n_processes) as pool:
            with tqdm.tqdm(total=len(path_list), desc="Loading GT data") as pbar:
                # Use apply_async to allow progress bar updates
                results = [pool.apply_async(worker, args=(path,), callback=update_progress) for path in path_list]

                # Collect results once all processes are done
                scenarios = [r.get() for r in results]  # Ensures tasks complete before returning

        return cls(scenarios, False)

    @classmethod
    def from_folder(cls,
                    folder_path,
                    process=True,
                    parallel=True,
                    ):
        """
        Loads a dataset from a folder containing multiple simulation files.
        The files must contain in the name 'simulation' and end with '.pkl'.

        :param folder_path: Path to the folder
        :param parallel: If True, the datasets will be loaded in parallel
        :return: A GroundTruthDataset object
        """
        # Get list of files
        files = [f for f in folder_path.iterdir() if f.is_file() and "simulation" in f.name and f.suffix == ".pkl"]

        # Load scenarios
        return cls.load_list(files, process, parallel)

    def extract(self, n=1):
        """
        Extracts n scenarios from the dataset and returns the processed data for the simulation
        :param n: Number of scenarios to extract
        :return: A list of dictionaries, each containing the processed data for the simulation
        """
        # If n is greater than the number of scenarios, return all the scenarios
        n = min(n, len(self.scenarios)) if n is not None else len(self.scenarios)

        # Shuffle and pop n scenarios
        random.shuffle(self.scenarios)
        extracted_scenarios = [self.scenarios.pop(0) for _ in range(n)]

        return extracted_scenarios

    def get_batches(self, n=1):
        """
        Extracts n scenarios from the dataset and returns the processed data for the simulation
        :param n: Number of scenarios to extract
        :return: A list of dictionaries, each containing the processed data for the simulation
        """
        batches = []
        while len(self.scenarios) > 0:
            batch = self.extract(n)
            batches.append(batch)

        return batches


"""
UTILITY FUNCTIONS
"""


def load(path,
         origin_folder=None,
         parallel=True,
         process_data=True,
         skip_if_found=True,
         ):
    path = Path(path)

    if path.exists() and skip_if_found:
        gt_data = GroundTruthDataset.load(path)

    elif origin_folder is not None:
        gt_data = GroundTruthDataset.from_folder(origin_folder, process=process_data, parallel=parallel)
        save(gt_data, path)
        print(f"Ground truth data saved to {path}")

    else:
        raise FileNotFoundError("The file does not exist and no origin folder was provided.")

    return gt_data


def save(gt_data, path):
    with open(path, "wb") as f:
        pickle.dump(gt_data, f)
