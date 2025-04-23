"""
A class that implements different approaches for selecting the best GBM alternatives.
In general, simulations are cached in the disk and then, whenever needed, reused to reduce computational burden of
experiments.

Author: Juan-Pablo Futalef
"""

import copy
import os
import random
from typing import Union, Dict

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
from greyboxmodels.construction import BayesianMetric
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

    mean_result = np.mean(results, dtype=np.float64)
    variance_result = np.var(results, dtype=np.float64)
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


def loss(mu_L1,
         mu_L2,
         sigma_L1=0.0,
         sigma_L2=0.0,
         w1=0.5,
         w2=0.5,
         lambda1=1.0,
         lambda2=1.0,
         ):
    """
    Compute the loss for a given substitution plan.

    :param mu_L1: Mean computational load for the plan.
    :param mu_L2: Mean fidelity for the plan.
    :param sigma_L1: Variance associated with the computational load.
    :param sigma_L2: Variance associated with the fidelity.
    :param w1: Weight factor for the computational load component.
    :param w2: Weight factor for the lack of fit component.
    :param lambda1: Scaling factor for the computational load.
    :param lambda2: Scaling factor for the lack of fit.
    :param add_sigma: Boolean flag indicating whether to include sigma in the loss computation.
    :return: The computed loss value.
    """
    # Obtain the main loss
    mu_L = w1 * lambda1 * mu_L1 + w2 * lambda2 * mu_L2
    sigma_L = np.sqrt((lambda1 * sigma_L1) ** 2 + (lambda2 * sigma_L2) ** 2)

    L = mu_L + sigma_L

    return L


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
        model = repository.get_model(plan)
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
    plan_simulations = SimulationDataset.SimulationDataset(plan_simulations)

    return plan, plan_simulations


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
            "voi_driven": self._pseudo_random_heuristic,
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

    def _exhaustive_optimizer(self,
                              n_scenarios=None,
                              parallel_plans=True,
                              parallel_scenarios=True,
                              plans=None,
                              ):
        """
        Exhaustively simulates substitution plans and selects the one with the highest VoI.
        
        :param n_scenarios: Limit on the number of scenarios to simulate (None uses all scenarios).
        :param parallel_plans: Run simulation of plans in parallel using multiprocessing.
        :param parallel_scenarios: Run simulation of scenarios in parallel within each plan.
        :param plans: List of substitution plans to evaluate (None uses all available plans).
        :return: Tuple (best_plan, performance metrics dictionary)
        """
        """
        INITIALIZATION
        """
        # Create necessary directories
        os.makedirs(self.work_dir, exist_ok=True)  # Main working directory
        os.makedirs(self.work_dir / "simulations", exist_ok=True)  # Directory for simulation results

        # Get references
        ref_scenarios = self.gt_data.extract(n_scenarios) if n_scenarios is not None else self.gt_data.scenarios
        ref_model = self.repository.get_model(self._ref_plan)

        # If no specific plans are provided, use all available plans.
        if plans is None:
            plans = self.substitution_plans

        """
        SIMULATION OF ALL PLANS AND SCENARIOS
        """
        # Simulate the scenarios (THIS IS THE EXPENSIVE PART!!)
        simulation_results = simulate_several_plans(plans, ref_scenarios, self.repository, self.work_dir,
                                                    parallel_plans, parallel_scenarios)

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

        """
        COMPUTATION OF LOSSESS AND VOI
        """
        # Storage for results
        loss_dict = {}
        voi_dict = {}

        # Compute losses for the reference plan
        (mean_l1, var_l1), loss_info_l1 = computational_load(simulation_results[self._ref_plan].scenarios)
        (mean_l2, var_l2), loss_info_l2 = lack_of_fit(ref_scenarios,
                                                      simulation_results[self._ref_plan].scenarios,
                                                      self._risk_metric,
                                                      ref_model,
                                                      ref_model,
                                                      )
        loss_ref = loss(mean_l1,
                        mean_l2,
                        sigma_L1=var_l1,
                        sigma_L2=var_l2,
                        w1=self._w1,
                        w2=self._w2,
                        lambda1=self._lambda1,
                        lambda2=self._lambda2,
                        )

        # Store the ref loss information
        loss_dict["ref"] = {"mu_l1": mean_l1,
                            "sigma_l1": var_l1,
                            "mu_l2": mean_l2,
                            "sigma_l2": var_l2,
                            "loss": loss_ref,
                            }

        # Evaluate performance of each plan to compute actual VoI
        for plan in plans:
            # Get scenarios and model
            scenarios = simulation_results[plan].scenarios
            model = self.repository.get_model(plan)

            # Compute the loss for those scenarios
            (mean_l1, var_l1), loss_info_l1 = computational_load(scenarios)
            (mean_l2, var_l2), loss_info_l2 = lack_of_fit(ref_scenarios,
                                                          scenarios,
                                                          self._risk_metric,
                                                          ref_model,
                                                          model,
                                                          )
            # Compute the loss for the reference plan
            loss_plan = loss(mean_l1,
                             mean_l2,
                             sigma_L1=var_l1,
                             sigma_L2=var_l2,
                             w1=self._w1,
                             w2=self._w2,
                             lambda1=self._lambda1,
                             lambda2=self._lambda2,
                             )
            voi = loss_ref - loss_plan

            # Store the information
            loss_dict[plan] = {
                "mu_l1": mean_l1,
                "sigma_l1": var_l1,
                "mu_l2": mean_l2,
                "sigma_l2": var_l2,
                "loss": loss_plan,
            }
            voi_dict[plan] = voi

        # Create a dataframe to store the results
        loss_df = pd.DataFrame.from_dict(loss_dict, orient="index")
        loss_df.index = loss_df.index.map(lambda x: self._plan_names.get(x, x))
        loss_df.to_csv(self.work_dir / "loss_summary.csv")

        voi_df = pd.DataFrame.from_dict(voi_dict, orient="index")
        voi_df.index = voi_df.index.map(lambda x: self._plan_names.get(x, x))
        voi_df.to_csv(self.work_dir / "voi_summary.csv")

        # Return the best plan and performance information
        best_plan = max(voi_dict, key=lambda p: voi_dict[p])
        info = {"voi": voi_dict,
                "loss_ref": loss_ref,
                }
        return best_plan, info

    def _pseudo_random_heuristic(self,
                                 prior_l1: Dict[tuple, tuple],
                                 prior_l2: Dict[tuple, tuple],
                                 n_out=3,
                                 n_batch=5,
                                 parallel_scenarios=False,
                                 plans=None,
                                 ):
        """
        Uses a pseudo-random greedy heuristic to select the next substitution plan.
        
        :param n_out: Number of recent plans to exclude.
        :param n_batch: Number of batches to simulate.
        :param save_in: Path to save temporary results/history.
        :return: Tuple (best_plan, performance info)
        """

        def next_plan_heuristic(l1_repo, l2_repo, out_plans, n_out):
            # Get prior values
            return

        """
        INITIALIZE PROCEDURE
        """
        # Create necessary directories
        os.makedirs(self.work_dir, exist_ok=True)  # Main working directory
        os.makedirs(self.work_dir / "simulations", exist_ok=True)  # Directory for simulation results

        # If no specific plans are provided, use all available plans for the construction
        if plans is None:
            plans = self.substitution_plans

        # Remove reference plan from the list of plans
        plans.remove(self._ref_plan)

        # metadata of the procedure
        metadata = {"parallel_scenarios": parallel_scenarios,
                    "n_out": n_out,
                    "n_batch": n_batch,
                    "method": "pseudo_random_heuristic",
                    "w1": self._w1,
                    "w2": self._w2,
                    "lambda1": self._lambda1,
                    "lambda2": self._lambda2,
                    "prior_l1": prior_l1,
                    "prior_l2": prior_l2,
                    }

        """
        PRIOR ESTIMATIONS
        """
        Estimator = BayesianMetric.BayesianNormalEstimator
        l1_estimator_repo = {plan: Estimator(*prior_l1[plan]) for plan in plans}
        l2_estimator_repo = {plan: Estimator(*prior_l2[plan]) for plan in plans}

        """
        REFERENCE LOSS
        """
        # Compute losses for the reference plan
        (mean_l1, var_l1), loss_info_l1 = computational_load(self.gt_data.scenarios)
        (mean_l2, var_l2), loss_info_l2 = lack_of_fit(self.gt_data.scenarios,
                                                      self.gt_data.scenarios,
                                                      self._risk_metric,
                                                      self.repository.get_model(self._ref_plan),
                                                      self.repository.get_model(self._ref_plan),
                                                      )
        loss_ref = loss(mean_l1,
                        mean_l2,
                        sigma_L1=var_l1,
                        sigma_L2=var_l2,
                        w1=self._w1,
                        w2=self._w2,
                        lambda1=self._lambda1,
                        lambda2=self._lambda2,
                        )

        # Store the ref loss information
        metadata["ref_loss"] = {"mu_l1": mean_l1,
                                "sigma_l1": var_l1,
                                "mu_l2": mean_l2,
                                "sigma_l2": var_l2,
                                "loss": loss_ref,
                                }

        """
        GET BATCHES
        """
        batches = self.gt_data.get_batches(n_batch)

        """
        MAIN PROCEDURE
        """
        out_plans = {}  # A container for the plans temporarily out of the game
        selection_info = {"losses": {plan: [] for plan in plans},
                          "voi": {plan: [] for plan in plans},
                          "chosen_plan": [],
                          "out_plans": [],
                          }

        for k, batch in enumerate(tqdm.tqdm(batches, desc="Simulating batches")):
            # Store out plans
            selection_info["out_plans"].append(copy.deepcopy(out_plans))

            # Compute priori VoI
            prior_voi = {}
            for plan in plans:
                # Compute elements
                l1_mean, l1_var = l1_estimator_repo[plan].get_mean_variance()
                l2_mean, l2_var = l2_estimator_repo[plan].get_mean_variance()
                l = loss(l1_mean, l2_mean, l1_var, l2_var, self._w1, self._w2, self._lambda1, self._lambda2, )
                voi = loss_ref - l

                # Store
                loss_dict = {
                    "mu_l1": l1_mean,
                    "sigma_l1": l1_var,
                    "mu_l2": l2_mean,
                    "sigma_l2": l2_var,
                    "loss": l,
                }
                selection_info["losses"][plan].append(loss_dict)
                selection_info["voi"][plan].append(voi)

                if plan in out_plans:
                    # Skip plans that are temporarily out of the game
                    continue

                prior_voi[plan] = voi

            # Create selection probabilities as roulette wheel using VoI
            voi_array = np.array(list(prior_voi.values()))
            off_voi_array = voi_array - np.min(voi_array)
            sum_voi = np.sum(off_voi_array)
            probabilities = off_voi_array / sum_voi

            # Select the next plan based on the probabilities
            current_plan = random.choices(population=list(prior_voi.keys()), weights=probabilities, k=1)[0]

            # Store
            selection_info["chosen_plan"].append(current_plan)

            # Simulate the batch
            plan, sim_data = simulate_plan(current_plan, batch, self.repository, self.work_dir, parallel_scenarios)

            # Collect evidence from the simulations

            (mean_l1, var_l1), loss_info_l1 = computational_load(sim_data.scenarios)
            (mean_l2, var_l2), loss_info_l2 = lack_of_fit(batch,
                                                          sim_data.scenarios,
                                                          self._risk_metric,
                                                          self.repository.get_model(current_plan),
                                                          self.repository.get_model(self._ref_plan),
                                                          )

            # Update computational with collected evidence
            l1_evidence = [x['slope'] for x in loss_info_l1['details']]
            l1_estimator_repo[current_plan].update(l1_evidence)

            # Update lack of fit with collected evidence
            l2_evidence = loss_info_l2['ks_value']
            l2_estimator_repo[current_plan].update(l2_evidence)

            # Update the out_plans container
            for plan in list(out_plans.keys()):
                out_plans[plan] -= 1
                if out_plans[plan] <= 0:
                    del out_plans[plan]

                    # Include the current plan in the out_plans container
            out_plans[current_plan] = n_out

        # Compute voi using the last updates
        post_voi = {}
        for plan in plans:
            # Compute elements
            l1_mean, l1_var = l1_estimator_repo[plan].get_mean_variance()
            l2_mean, l2_var = l2_estimator_repo[plan].get_mean_variance()
            l = loss(l1_mean, l2_mean, l1_var, l2_var, self._w1, self._w2, self._lambda1, self._lambda2, )
            voi = loss_ref - l

            post_voi[plan] = voi

        # Store the final VoI
        selection_info["final_voi_values"] = post_voi

        # Store the selection information in the metadata
        metadata["selection_info"] = selection_info

        # Save the metadata to a file
        metadata_path = self.work_dir / "procedure_details.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # Save selection information
        selection_info_path = self.work_dir / "selection_details.pkl"
        with open(selection_info_path, "wb") as f:
            pickle.dump(selection_info, f)

        # Create a dataframe of the final results
        final_voi_df = pd.DataFrame.from_dict(post_voi, orient="index", columns=["final_voi"])
        final_voi_df.index = final_voi_df.index.map(lambda x: self._plan_names.get(x, x))
        final_voi_df.to_csv(self.work_dir / "final_voi_summary.csv")

        # Get the best plan
        best_plan = max(post_voi, key=lambda p: post_voi[p])

        return best_plan, metadata

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
