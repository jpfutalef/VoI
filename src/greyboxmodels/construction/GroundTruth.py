"""
A class that represents a dataset of ground truth data and implements several methods to access and manipulate it.
The dataset is supposed to exclusively be used for the construction of grey-box models.

Author: Juan-Pablo Futalef
"""
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


def process_scenario(scenario):
    """
    Extracts the relevant data from one scenario
    :param scenario: A dictionary containing the simulation data
    :return: A dictionary containing the relevant data for the simulation
    """
    # Time
    t = scenario["time"]

    # Uncontrolled inputs
    e = scenario["uncontrolled_inputs"]
    e = Input.Input(pd.DataFrame(e, index=t))

    # State variables
    x = scenario["state"]
    x0 = scenario["initial_state"]
    x = np.vstack([x0, x])
    t0 = scenario["initial_time"]
    t = np.hstack([t0, t])
    x = Input.Input(pd.DataFrame(x, index=t))

    # Return the relevant data
    return {"initial_time": t0,
            "mission_time": scenario["mission_time"],
            "time_step": t[1] - t[0],
            "initial_state": x0,
            "external_stimuli": e,
            "forced_states": x,
            }


def load_process_simulation_file(path):
    sim_data = Simulator.load_simulation_data(path)
    s = process_scenario(sim_data)
    return s


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
                  parallel=True,
                  ):
        """
        Loads a list of datasets from a list of files.

        :param path_list: List of paths to the files
        :return: A list of GroundTruthDataset objects
        """
        if not parallel:
            return cls([load_process_simulation_file(path) for path in path_list], False)

        # Use multiprocessing to load datasets in parallel
        n_processes = min(mp.cpu_count(), len(path_list))  # Don't create more processes than needed

        def update_progress(_):
            """Callback function to update tqdm progress bar."""
            pbar.update(1)

        with mp.Pool(n_processes) as pool:
            with tqdm.tqdm(total=len(path_list), desc="Loading datasets") as pbar:
                # Use apply_async to allow progress bar updates
                results = [pool.apply_async(load_process_simulation_file, args=(path,), callback=update_progress) for
                           path in path_list]

                # Collect results once all processes are done
                scenarios = [r.get() for r in results]  # Ensures tasks complete before returning

        return cls(scenarios, False)

    @classmethod
    def from_folder(cls,
                    folder_path,
                    parallel=True,
                    ):
        """
        Loads a dataset from a folder containing multiple simulation files.
        The files must contain in the name 'simulation' and end with '.pkl'.

        :param folder_path: Path to the folder
        :return: A GroundTruthDataset object
        """
        # Get list of files
        files = [f for f in folder_path.iterdir() if f.is_file() and "simulation" in f.name and f.suffix == ".pkl"]

        # Load scenarios
        return cls.load_list(files, parallel)

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
         process_if_not_found=False
         ):
    path = Path(path)

    try:
        gt_data = GroundTruthDataset.load(path)
    except FileNotFoundError:   # TODO verify integrity when handling this exception, what if the files in folder are not correct?
        if not process_if_not_found:
            m = f"Ground truth not found at: {path}\n   (process_if_not_found={process_if_not_found})"
            raise FileNotFoundError(m)
        gt_data = GroundTruthDataset.from_folder(path.parent, parallel=True)
        save(gt_data, path)

    return gt_data


def save(gt_data, path):
    with open(path, "wb") as f:
        pickle.dump(gt_data, f)
