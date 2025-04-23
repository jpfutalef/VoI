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


def load_and_process_simulation_file(path, process_fun):
    s = Simulator.load_simulation_data(path)
    try:
        s = process_fun(s)
    except Exception as e:
        print(f"Error processing scenario: {path}")
        print(e)
        return None
    return s


def default_process_scenario(scenario):
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
        processed_scenario = default_process_scenario(scenario)
        processed_scenarios.append(processed_scenario)

    return processed_scenarios


class SimulationDataset:
    def __init__(self,
                 scenario_list: List[dict],
                 scenario_id: List[str] = None,
                 ):
        """
            Initializes the dataset.

            :param scenario_list: A list of dictionaries, each containing an scenario of simulation data.
            """
        self.scenarios = scenario_list
        self.scenario_id = scenario_id

    @classmethod
    def load(cls, path):
        """
        Loads a SimulationDataset object from a file.

        :param path: Path to the file
        :return: A SimulationDataset object
        """
        with open(path, "rb") as file:
            instance = pickle.load(file)

        return instance

    @classmethod
    def load_list(cls,
                  path_list,
                  remove_step_data=True,
                  parallel=True,
                  ):
        """
        Loads a list of datasets from a list of files.

        :param path_list: List of paths to the files
        :return: A list of SimulationDataset objects
        """
        worker = load_and_process_simulation_file if remove_step_data else Simulator.load_simulation_data
        worker_args = [(path, default_process_scenario) if remove_step_data else (path,) for path in path_list]

        if parallel:
            # Use multiprocessing to load datasets in parallel
            def update_progress(_):
                """Callback function to update tqdm progress bar."""
                pbar.update(1)

            n_processes = min(mp.cpu_count(), len(path_list))
            with mp.Pool(n_processes) as pool:
                with tqdm.tqdm(total=len(path_list), desc="Loading GT data (parallel)") as pbar:
                    # Use apply_async to allow progress bar updates
                    results = [pool.apply_async(worker, args=args, callback=update_progress)
                               for args in worker_args]

                    # Collect results once all processes are done
                    scenarios = [r.get() for r in results]  # Ensures tasks complete before returning
        else:
            # Load scenarios sequentially
            scenarios = [worker(*args) for args in tqdm.tqdm(worker_args, desc="Loading GT data (sequentially)")]

        # Add the ids to each scenario
        ids = [path.stem for path in path_list]
        for i, scenario in enumerate(scenarios):
            scenario["id"] = ids[i]

        return cls(scenarios, ids)

    @classmethod
    def from_folder(cls,
                    folder_path,
                    remove_step_data=True,
                    parallel=True):
        """
        Loads a dataset from a folder containing multiple simulation files.
        The files must contain in the name 'simulation' and end with '.pkl'.

        :param folder_path: Path to the folder
        :param remove_step_data: If True, the scenarios will be processed to remove step data
        :param parallel: If True, the datasets will be loaded in parallel
        :return: A SimulationDataset object
        """
        files = [f for f in folder_path.iterdir() if f.is_file() and "simulation" in f.name and f.suffix == ".pkl"]
        return cls.load_list(path_list=files, remove_step_data=remove_step_data, parallel=parallel)

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

    def process_scenarios(self, process_fun):
        """
        Processes the scenarios in the dataset using the given function.
        :param process_fun: Function to process the scenarios
        :return: A SimulationDataset object with the processed scenarios
        """
        new_scenarios = []
        for i, scenario in tqdm.tqdm(enumerate(self.scenarios),
                                     total=len(self.scenarios),
                                     desc="Processing scenarios"):
            processed_scenario = process_fun(scenario)
            new_scenarios.append(processed_scenario)

        return SimulationDataset(new_scenarios, self.scenario_id)

    def get_scenario_by_index(self, index):
        """
        Retrieves a scenario based on its index.

        :param index: Index of the scenario to retrieve.
        :return: The scenario at the specified index.
        """
        if index < 0 or index >= len(self.scenarios):
            raise IndexError("Index out of range.")
        return self.scenarios[index]

    def get_scenario_by_id(self, scenario_id):
        """
        Retrieves a scenario based on its ID.

        :param scenario_id: ID of the scenario to retrieve.
        :return: The scenario with the specified ID.
        """
        if self.scenario_id is None:
            raise ValueError("Scenario IDs are not available.")
        if scenario_id not in self.scenario_id:
            raise ValueError("Scenario ID not found.")
        index = self.scenario_id.index(scenario_id)
        return self.scenarios[index]

"""
UTILITY FUNCTIONS
"""


def load(path):
    try:
        gt_data = SimulationDataset.load(path)
        return gt_data

    except FileNotFoundError:
        raise FileNotFoundError("The file does not exist and no origin folder was provided.")


def save(gt_data, path):
    with open(path, "wb") as f:
        pickle.dump(gt_data, f)
