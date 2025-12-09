import copy
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import dill as pickle
import numpy as np
import pandas as pd
import tqdm

import greyboxmodels.construction.SimulationDataset as SimulationDataset
from greyboxmodels.modelbuild import Input
from greyboxmodels.simulation import Simulator


def simulate_scenario(scenario, model, substitution_plan, position):
    """
    Simulates a single scenario using the given grey-box model.

    :param scenario: Dictionary with scenario data.
    :param model: Model instance.
    :param substitution_plan: The plan identifier.
    :param position: Position index for progress reporting.
    :return: Simulation data dictionary (or None if simulation fails).
    """
    try:
        t = scenario["time"]
        mission_time = scenario["mission_time"]
        dt = t[1] - t[0]
        e = Input.Input(pd.DataFrame(scenario["uncontrolled_inputs"], index=t))
        x = np.vstack([scenario["initial_state"], scenario["state"]])
        t_full = np.hstack([scenario["initial_time"], t])
        x_input = Input.Input(pd.DataFrame(x, index=t_full))
        params = Simulator.SimulationParameters(
            initial_time=scenario["initial_time"],
            mission_time=mission_time,
            time_step=dt,
            initial_state=scenario["initial_state"],
            external_stimuli=e,
            forced_states=x_input,
        )
        simulator = Simulator.Simulator(model, params)
        sim_data = simulator.simulate(pbar_post=f"S={substitution_plan} | Scenario: {scenario['id']}",
                                      pbar_position=position)
        sim_data["id"] = scenario["id"]
        return sim_data

    except Exception as e:
        tqdm.tqdm.write(f"Failed ({e}): S={substitution_plan} | Scenario: {scenario['id']}")
        return None


def simulate_plan(plan, scenarios, repository, work_dir, parallel=True, pbar_offset=0):
    """
    Simulates all scenarios for a given substitution plan with caching.

    :param plan: The substitution plan identifier.
    :param scenarios: List of scenario dictionaries.
    :param repository: Repository instance providing the grey-box model.
    :param work_dir: Working directory path for caching simulation results.
    :param parallel: Boolean flag to run scenario simulations in parallel.
    :param pbar_offset: Offset for progress bar positioning.
    :return: Tuple (plan, simulation results)
    """
    simulation_path = work_dir / "simulations" / plan_filename(plan)
    available_simulations = []
    try:
        # Check if the simulation results are already cached
        with open(simulation_path, "rb") as f:
            cached_simulations = pickle.load(f)

        # Extend the available simulations with the cached ones
        available_simulations.extend(cached_simulations.scenarios)

        # If so, check the scenarios that are already cached
        cached_ids = {sim["id"] for sim in cached_simulations.scenarios}

        # Store the scenarios that are not cached for later simulations
        pending_scenarios = [sc for sc in scenarios if sc["id"] not in cached_ids]

    except FileNotFoundError:
        # Simulate all scenarios if the file is not found
        pending_scenarios = scenarios

    if pending_scenarios:
        # Simulate all pending scenarios
        model = repository[plan]
        model.stochastic = False  # Ensure deterministic simulation
        results = []
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = []
                for i, scenario in enumerate(pending_scenarios):
                    model_copy = copy.deepcopy(model)
                    futures.append(executor.submit(simulate_scenario, scenario, model_copy, plan, pbar_offset + i + 1))
                results = [future.result() for future in
                           tqdm.tqdm(futures, desc="Simulating scenarios", position=pbar_offset)]
        else:
            for i, scenario in enumerate(
                    tqdm.tqdm(pending_scenarios, desc="Simulating scenarios", position=pbar_offset)):
                result = simulate_scenario(scenario, model, plan, pbar_offset + i + 1)
                results.append(result)

        # Extend the available simulations with the new results
        available_simulations.extend(results)

        # Turn into a SimulationDataset
        available_simulations = SimulationDataset.SimulationDataset(available_simulations)

        # Save the new simulations to the cache
        with open(simulation_path, "wb") as f:
            pickle.dump(available_simulations, f)

    # Create the output by retrieving from the available simulations the ids of the scenarios
    requested_scenarios = {sc["id"] for sc in scenarios}
    plan_simulations = [sim for sim in available_simulations if sim["id"] in requested_scenarios]

    return plan_simulations


def simulate_several_plans(plans, ref_scenarios, repository, work_dir, parallel_plans, parallel_scenarios):
    """
    Simulates several substitution plans to some reference scenarios.

    :param plans: List of substitution plans to evaluate.
    :param ref_scenarios: List of reference scenarios.
    :param repository: Instance of GreyBoxRepository for model retrieval.
    :param work_dir: Path to the working directory for storing simulation results.
    :param parallel_plans: Boolean flag to enable parallel execution of plan simulations.
    :param parallel_scenarios: Boolean flag to enable parallel execution of scenario simulations within each plan.
    :return: Dictionary mapping each substitution plan to its corresponding simulation results.
    """
    simulation_results = {}
    if parallel_plans:
        args_list = []
        for plan in plans:
            args_list.append((plan, ref_scenarios, repository, work_dir, parallel_scenarios))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(simulate_plan, args_list)
        simulation_results = {plan: sim_data for plan, sim_data in results}
    else:
        for plan in plans:
            plan, sim_data = simulate_plan(plan, ref_scenarios, repository, work_dir, parallel=parallel_scenarios)
            simulation_results[plan] = sim_data
    return simulation_results


def total_execution_time(simulation_results):
    """
    Computes the total execution time for a list of simulation results.

    :param simulation_results: Dictionary of simulation results.
    :return: Total execution time.
    """
    total_time = 0
    for plan, sim_dataset in simulation_results.items():
        for scenario in sim_dataset.scenarios:
            total_time += scenario['total_execution_time']
    return total_time


def plan_filename(plan):
    # Generate a filename like this: "plan_0-1-0-0.pkl"
    plans_str = "-".join(map(str, plan))
    return f"{plans_str}.pkl"
