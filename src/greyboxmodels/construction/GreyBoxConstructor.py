#!/usr/bin/env python3
"""
Module: greybox_model_constructor.py
Description:
    Optimized implementation of a GreyBoxModelConstructor class that iteratively constructs and optimizes greybox models.
    This version unifies simulation routines for parallel/sequential processing and offloads loss computation 
    to a separate module (loss.py). Functions that are used in parallel processing are defined at module level 
    to ensure picklability.

Usage:
    # Construct a GreyBox model
    from greybox_model_constructor import GreyBoxModelConstructor
    from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository
    import greyboxmodels.construction.SimulationDataset as SimulationDataset

    # (Initialize your repository, ground truth data, and risk metric appropriately)
    repository = GreyBoxRepository(...)
    gt_data = SimulationDataset.SimulationDataset(...)
    risk_metric = lambda sim_data, plant: ...  # Your risk metric function

    constructor = GreyBoxModelConstructor(repository, gt_data, risk_metric, work_dir="path/to/work_dir")
    best_plan, performance = constructor.construct(method="exhaustive_numerical")
"""

import copy
import os
import random
from typing import Union

import numpy as np
import pandas as pd
from pathlib import Path
import dill as pickle
import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Import simulation components from your package:
from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository
import greyboxmodels.construction.SimulationDataset as SimulationDataset
from greyboxmodels.simulation import Simulator
from greyboxmodels.modelbuild import Input
from greyboxmodels.modelbuild.Plant import Plant


# -----------------------------------------------------------------------------
# Utility Functions (Computational Load and Lack of Fit)
# -----------------------------------------------------------------------------

def computational_load(sim_data_list):
    """
    Compute the computational load for a list of simulation data dictionaries.
    
    :param sim_data_list: List of dictionaries with simulation data.
    :return: Tuple ((mean, variance), detailed_info)
    """

    def slope_fit(t_sim, t_exec):
        t_sim_col = t_sim[:, np.newaxis]
        m, _, _, _ = np.linalg.lstsq(t_sim_col, t_exec, rcond=None)
        return m[0]

    results = []
    info = []
    for sim_data in sim_data_list:
        t_sim = np.array(sim_data["time"])
        t_exec = np.array(sim_data["execution_time_array"]) - sim_data["execution_time_array"][0]
        m = slope_fit(t_sim, t_exec)
        results.append(m)
        info.append({"t_sim": t_sim, "t_exec": t_exec, "slope": m})

    mean_result = np.mean(results)
    variance_result = np.var(results)
    info_dict = {"mean": mean_result, "variance": variance_result, "details": info}
    return (mean_result, variance_result), info_dict


def lack_of_fit(ref_sim_data_list, sim_data_list, risk_metric, plant_ref, plant_gbm):
    """
    Compute the lack of fit between two simulation datasets using a risk metric.
    
    :param ref_sim_data_list: List of reference simulation data dictionaries.
    :param sim_data_list: List of simulation data dictionaries from the grey-box model.
    :param risk_metric: Callable risk metric used for comparing simulations.
    :param plant_ref: Reference plant model.
    :param plant_gbm: Grey-box model.
    :return: Tuple ((mean, variance), detailed_info)
    """

    def ks_statistic(data_ref, data, n_bins=20):
        if np.array_equal(data_ref, data):
            return 0, {"equal_data": True}
        min_x = min(data.min(), data_ref.min())
        max_x = max(data.max(), data_ref.max())
        bins = np.linspace(min_x, max_x, n_bins + 1)
        epdf_ref, _ = np.histogram(data_ref, bins=bins, density=True)
        epdf, _ = np.histogram(data, bins=bins, density=True)
        ecdf_ref = empirical_cdf(bins, epdf_ref)
        ecdf = empirical_cdf(bins, epdf)
        abs_diff = np.abs(ecdf - ecdf_ref)
        ks_value = np.max(abs_diff)
        max_diff_idx = np.argmax(abs_diff)
        info = {"ks_idx": max_diff_idx, "ks_value": ks_value, "ks_bin_loc": bins[max_diff_idx],
                "bins": bins, "epdf": epdf, "epdf_ref": epdf_ref, "ecdf": ecdf, "ecdf_ref": ecdf_ref,
                "abs_diff": abs_diff}
        return ks_value, info

    def empirical_cdf(bin_edges, epdf):
        cdf_values = np.cumsum(epdf * np.diff(bin_edges))
        cdf_values /= cdf_values[-1]
        return np.concatenate(([0], cdf_values))

    def aggregate(t, r):
        return np.sum(r) / len(r)

    metric_ref = [risk_metric(ref, plant_ref) for ref in ref_sim_data_list]
    metric = [risk_metric(sim, plant_gbm) for sim in sim_data_list]
    agg_metric_ref = np.array([aggregate(x["time"], m) for x, m in zip(ref_sim_data_list, metric_ref)])
    agg_metric = np.array([aggregate(x["time"], m) for x, m in zip(sim_data_list, metric)])
    ks, info_dict = ks_statistic(agg_metric_ref, agg_metric)
    mean_result = ks
    variance_result = ks * 0.05 + 1e-6
    return (mean_result, variance_result), info_dict


def loss(scenarios: list,
         ref_scenarios: list,
         model: Plant,
         ref_model: Plant,
         risk_metric: callable,
         w1=0.5,
         w2=0.5,
         lambda1=1.0,
         lambda2=1.0,
         add_sigma=True,
         ):
    """
    Compute the loss for a given substitution plan.

    :param scenarios: SimulationDataset object containing simulated scenarios.
    :param ref_scenarios: SimulationDataset object containing reference scenarios.
    :param model: Grey-box model instance.
    :param ref_model: Reference model instance.
    :param risk_metric: Callable risk metric function.
    :param w1: Weight for computational load.
    :param w2: Weight for lack of fit.
    :param lambda1: Scaling factor for computational load.
    :param lambda2: Scaling factor for lack of fit.
    :param add_sigma: If true, add sigma when computational load.
    :return: Tuple (loss, info_dict)
    """
    # Compute the individual losses
    (mu_L1, sigma_L1), info_l1 = computational_load(scenarios)
    (mu_L2, sigma_L2), info_l2 = lack_of_fit(ref_scenarios, scenarios, risk_metric, ref_model, model)

    # Store results in a dict
    info = {"computational_load_info": info_l1,
            "lack_of_fit_info": info_l2,
            "mu_l1": mu_L1, "sigma_l1": sigma_L1,
            "mu_l2": mu_L2, "sigma_l2": sigma_L2,
            }

    # Obtain the main loss
    mu_L = w1 * lambda1 * mu_L1 + w2 * lambda2 * mu_L2
    sigma_L = np.sqrt((lambda1 * sigma_L1) ** 2 + (lambda2 * sigma_L2) ** 2) if add_sigma else 0.

    L = mu_L + sigma_L

    return L, info


# -----------------------------------------------------------------------------
# Simulation Helper Functions
# -----------------------------------------------------------------------------

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


def simulate_scenarios_for_plan(plan, scenarios, repository, parallel=True, pbar_offset=0):
    """
    Simulates all scenarios for a given substitution plan.

    :param plan: The substitution plan identifier.
    :param scenarios: List of scenario dictionaries.
    :param repository: Repository instance providing the grey-box model.
    :param parallel: Boolean flag to run scenario simulations in parallel.
    :param pbar_offset: Offset for progress bar positioning.
    :return: List of simulation results.
    """
    model = repository.get_model(plan)
    model.stochastic = False  # Ensure deterministic simulation
    results = []
    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, scenario in enumerate(scenarios):
                model_copy = copy.deepcopy(model)
                futures.append(executor.submit(simulate_scenario, scenario, model_copy, plan, pbar_offset + i + 1))
            results = [future.result() for future in
                       tqdm.tqdm(futures, desc="Simulating scenarios", position=pbar_offset)]

    else:
        for i, scenario in enumerate(tqdm.tqdm(scenarios, desc="Simulating scenarios", position=pbar_offset)):
            result = simulate_scenario(scenario, model, plan, pbar_offset + i + 1)
            results.append(result)

    return results


def simulate_plan_worker(args):
    """
    Worker function for multiprocessing. Simulates scenarios for a given plan,
    handling caching of results.
    
    :param args: Tuple containing (plan, scenarios, repository, work_dir, parallel flag)
    :return: Tuple (plan, simulation results)
    """

    # Unpack arguments
    plan, scenarios, repository, work_dir, parallel = args

    # Check if the simulation results are already cached
    simulation_path = work_dir / "simulations" / plan_filename(plan)
    try:
        with open(simulation_path, "rb") as f:
            plan_simulations = pickle.load(f)
    except FileNotFoundError:
        plan_simulations = simulate_scenarios_for_plan(plan, scenarios, repository, parallel=parallel)
        with open(simulation_path, "wb") as f:
            pickle.dump(plan_simulations, f)

    # Return the plan and its simulation results
    return plan, plan_simulations


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


# -----------------------------------------------------------------------------
# Main Class: GreyBoxModelConstructor
# -----------------------------------------------------------------------------

class GreyBoxModelConstructor:
    """
    Class that constructs and optimizes greybox models using iterative simulation and loss evaluation.
    """

    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: SimulationDataset.SimulationDataset,
                 risk_metric: callable,
                 work_dir: Union[Path, str],
                 ref_plan=None,
                 w1: float = 0.5,
                 w2: float = 0.5,
                 lambda1: float = 1.0,
                 lambda2: float = 1.0,
                 ):
        """
        Initialize the constructor.
        
        :param model_repository: Instance of GreyBoxRepository.
        :param gt_data: Ground truth simulation data.
        :param risk_metric: Callable risk metric function.
        :param work_dir: Working directory path to store simulation results.
        :param ref_plan: Reference plan; if not provided, the first available plan is used.
        :param ref_data: Optional reference simulation data.
        """
        self.repository = model_repository
        self.gt_data = gt_data
        self.work_dir = Path(work_dir)
        self.substitution_plans = list(self.repository.model_repository.keys())
        self._ref_plan = ref_plan if ref_plan is not None else self.substitution_plans[0]
        self._risk_metric = risk_metric
        self._plan_names = {s: f"s_{i}" for i, s in enumerate(self.substitution_plans)}

        # Default weights and lambda values (these are now embedded in the loss functions)
        self._w1 = w1
        self._w2 = w2
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._s0 = None  # Initial substitution plan (if provided externally)
        self.gt_batch_size = 10  # Default ground truth batch size

    def construct(self, method="exhaustive_numerical", save_in=None, **kwargs):
        """
        Main method to construct the best greybox model.
        
        :param method: The optimization method ('exhaustive_numerical' or 'pseudo_random_heuristic').
        :param save_in: Path to save the optimization results.
        :param kwargs: Additional keyword arguments for the optimizer.
        :return: Tuple (best_plan, performance_info)
        """
        methods = {
            "exhaustive_numerical": self._exhaustive_optimizer,
            "pseudo_random_heuristic": self._pseudo_random_heuristic,
        }
        try:
            optimizer = methods[method]
        except KeyError:
            raise ValueError(f"Method '{method}' not available. Choose from: {list(methods.keys())}")

        best_plan, info = optimizer(**kwargs)

        if save_in is not None:
            result_dict = {"method": method, "result": best_plan, "info": info}
            save_path = Path(save_in) / f"{method}_results.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(result_dict, f)
            print(f"Saved results in: {save_path}")

        return best_plan, info

    def _exhaustive_optimizer(self, n_scenarios=None, parallel_plans=True, parallel_scenarios=True, plans=None):
        """
        Exhaustively simulates substitution plans and selects the one with the highest VoI.
        
        :param n_scenarios: Limit on the number of scenarios to simulate (None uses all scenarios).
        :param parallel_plans: Run simulation of plans in parallel using multiprocessing.
        :param parallel_scenarios: Run simulation of scenarios in parallel within each plan.
        :param plans: List of substitution plans to evaluate (None uses all available plans).
        :return: Tuple (best_plan, performance metrics dictionary)
        """
        # Create necessary directories
        os.makedirs(self.work_dir, exist_ok=True)  # Main working directory
        os.makedirs(self.work_dir / "simulations", exist_ok=True)  # Directory for simulation results

        # Extract scenarios if n_scenarios is specified
        ref_scenarios = self.gt_data.extract(n_scenarios) if n_scenarios is not None else self.gt_data.scenarios
        ref_model = self.repository.get_model(self._ref_plan)

        # If no specific plans are provided, use all available plans.
        if plans is None:
            plans = self.substitution_plans

        # Get the plan names
        plan_names = [self._plan_names[plan] for plan in plans]

        # Parallelize plan simulation if requested.
        if parallel_plans:
            args_list = []
            for plan in plans:
                args_list.append((plan, ref_scenarios, self.repository, self.work_dir, parallel_scenarios))
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(simulate_plan_worker, args_list)
            simulation_results = {plan: sim_data for plan, sim_data in results}

        else:
            simulation_results = {}
            for plan in plans:
                args = (plan, ref_scenarios, self.repository, self.work_dir, parallel_scenarios)
                plan, sim_data = simulate_plan_worker(args)
                simulation_results[plan] = sim_data

        # Compute the execution times and create a dataframe to store results.
        total_exec_time = total_execution_time(simulation_results)
        sim_params = {"parallel_plans": parallel_plans,
                      "parallel_scenarios": parallel_scenarios,
                      "total_execution_time_seconds": total_exec_time,
                      "total_execution_time_hours": total_exec_time / 3600,
                      "method": "exhaustive_numerical",
                      }
        sim_params_df = pd.DataFrame.from_dict(sim_params, orient="index", columns=["value"])
        sim_params_df.to_csv(self.work_dir / "simulations" / "summary.csv")

        # Compute the loss of the reference plan
        loss_ref, loss_ref_info = loss(simulation_results[self._ref_plan].scenarios,
                                       ref_scenarios,
                                       ref_model,
                                       ref_model,
                                       self._risk_metric,
                                       w1=self._w1,
                                       w2=self._w2,
                                       lambda1=self._lambda1,
                                       lambda2=self._lambda2,
                                       add_sigma=False,
                                       )

        # Evaluate performance of each plan to compute actual VoI
        voi_dict = {}
        loss_dict = {}
        for plan in plans:
            # Get scenarios and model
            scenarios = simulation_results[plan].scenarios
            model = self.repository.get_model(plan)

            # Compute the loss for those scenarios
            loss_plan, loss_info = loss(scenarios,
                                        ref_scenarios,
                                        model,
                                        ref_model,
                                        self._risk_metric,
                                        w1=self._w1,
                                        w2=self._w2,
                                        lambda1=self._lambda1,
                                        lambda2=self._lambda2,
                                        add_sigma=False,
                                        )
            voi = loss_ref - loss_plan
            voi_dict[plan] = voi

            # Store the loss information
            loss_dict[plan] = {"loss": loss_plan, }
            loss_dict[plan].update(loss_info)
            del loss_dict[plan]['computational_load_info']
            del loss_dict[plan]['lack_of_fit_info']

        # Create a dataframe to store the results
        voi_df = pd.DataFrame.from_dict(voi_dict, orient="index")
        voi_df.index = plan_names
        voi_df.to_csv(self.work_dir / "voi_summary.csv")

        # Store the loss information
        loss_dict["ref"] = {"loss": loss_ref}
        loss_dict["ref"].update(loss_ref_info)
        del loss_dict["ref"]['computational_load_info']
        del loss_dict["ref"]['lack_of_fit_info']

        loss_df = pd.DataFrame.from_dict(loss_dict, orient="index")
        loss_df.index = plan_names + ["ref"]
        loss_df.to_csv(self.work_dir / "loss_summary.csv")

        # Return the best plan and performance information
        best_plan = max(voi_dict, key=lambda p: voi_dict[p])
        info = {"voi": voi_dict,
                "loss_ref": loss_ref,
                }
        return best_plan, info

    def _pseudo_random_heuristic(self,
                                 n_out=3,
                                 n_batch=5,
                                 save_in=None,
                                 ):
        """
        Uses a pseudo-random greedy heuristic to select the next substitution plan.
        
        :param n_out: Number of recent plans to exclude.
        :param n_batch: Number of batches to simulate.
        :param save_in: Path to save temporary results/history.
        :return: Tuple (best_plan, performance info)
        """
        # Create necessary directories
        os.makedirs(self.work_dir, exist_ok=True)  # Main working directory
        os.makedirs(self.work_dir / "simulations", exist_ok=True)  # Directory for simulation results

        performance = {"method": "pseudo_random_heuristic"}
        self._prior_num_scenarios = 50
        self._prior_batch_size = 10
        batches = self.gt_data.get_batches(self.gt_batch_size)
        plan_history = []
        current_plan = self._s0 if self._s0 is not None else self._next_substitution_plan_heuristic(plan_history,
                                                                                                    n_out)
        for batch in tqdm.tqdm(batches, desc="Simulating batches"):
            gbm_batch = simulate_scenarios_for_plan(current_plan, batch, self.repository, parallel=True)
            self.repository.update_performance(plan=current_plan,
                                               sim_data_list=gbm_batch,
                                               gt_data_list=batch,
                                               risk_metric=self._risk_metric)
            plan_history.append(current_plan)
            current_plan = self._next_substitution_plan_heuristic(plan_history, n_out)
            if save_in is not None:
                history = self._get_history()
                history["plan_history"] = plan_history
                temp_path = Path(save_in) / f"_temp_history_pseudo_random.pkl"
                with open(temp_path, "wb") as f:
                    pickle.dump(history, f)

        voi = plan_loss(self.repository, w1=self._w1, w2=self._w2)
        best_plan = max(voi, key=lambda p: voi[p]["voi"])
        performance["best_plan"] = best_plan
        performance["last_voi"] = voi
        performance["history"] = self._get_history()
        if save_in is not None:
            result_path = Path(save_in) / f"pseudo_random_heuristic_results.pkl"
            with open(result_path, "wb") as f:
                pickle.dump(performance, f)
            print(f"Saved results in: {result_path}")
        return best_plan, performance

    def _next_substitution_plan_heuristic(self, plan_history, n_prev_plans):
        """
        Selects the next substitution plan, excluding recently used ones.
        
        :param plan_history: List of previously selected plans.
        :param n_prev_plans: Number of recent plans to exclude.
        :return: Next substitution plan.
        """
        if self._s0 is None:
            valid_plans = [p for p in self.substitution_plans if p]
            return random.choice(valid_plans)
        last_plans = plan_history[-n_prev_plans:] if len(plan_history) > n_prev_plans else plan_history
        voi = plan_loss_variance_penalized(self._ref_plan, self.repository, w1=self._w1, w2=1.0)
        eligible = {p: v for p, v in voi.items() if p not in last_plans}
        if not eligible:
            return random.choice(self.substitution_plans)
        voi_values = np.array([v["voi"] for v in eligible.values()])
        exp_voi = np.exp(voi_values - np.max(voi_values))
        probabilities = exp_voi / np.sum(exp_voi)
        selected_plan = random.choices(population=list(eligible.keys()), weights=probabilities, k=1)[0]
        return selected_plan

    def _get_history(self):
        """
        Retrieves performance history from the repository.
        
        :return: Dictionary with history details.
        """
        history_l1 = {p: self.repository.model_performance[p]["computational_load"].get_history() for p in
                      self.substitution_plans}
        history_l2 = {p: self.repository.model_performance[p]["lack_of_fit"].get_history() for p in
                      self.substitution_plans}
        return {"history_l1": history_l1, "history_l2": history_l2}

    def get_best_model(self):
        """
        (Stub) Return the best model based on performance criteria.
        Extend this as needed.
        """
        return None

    def get_history(self):
        """
        Return the performance history.
        """
        return self._get_history()


# -----------------------------------------------------------------------------
# Utility Functions to Save/Load the Constructor Object
# -----------------------------------------------------------------------------

def load_gc(path):
    """
    Load a GreyBoxModelConstructor instance from a file.
    
    :param path: File path of the saved instance.
    :return: GreyBoxModelConstructor object.
    """
    with open(path, "rb") as f:
        gc = pickle.load(f)
    return gc


def save_gc(gc, path):
    """
    Save a GreyBoxModelConstructor instance to a file.
    
    :param gc: GreyBoxModelConstructor instance.
    :param path: Destination file path.
    """
    with open(path, "wb") as f:
        pickle.dump(gc, f)

# End of greybox_model_constructor.py
