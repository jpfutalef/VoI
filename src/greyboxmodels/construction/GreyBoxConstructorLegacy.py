"""
Iterative method to guide GBM construction using VoI, coupled with some optimization algorithm.

Author: Juan-Pablo Futalef
"""
import copy
import os
import random

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool
import dill as pickle
import tqdm
import multiprocessing as mp
from pathlib import Path

from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository
import greyboxmodels.construction.SimulationDataset as SimulationDataset
from greyboxmodels.simulation import Simulator
from greyboxmodels.modelbuild import Input


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: SimulationDataset.SimulationDataset,
                 risk_metric: callable,
                 work_dir,
                 ref_plan=None,
                 ref_data: SimulationDataset.SimulationDataset =None,
                 ):
        """
        This constructor enables the iterative selection of the best model from a repository of models
        and a dataset of ground truth data. The selection is guided by the Value of Information (VoI) metric.

        :param model_repository: A repository of grey-box models
        :param gt_data: Ground truth data used for fidelity comparison
        :param risk_metric: A callable that computes the risk metric for fidelity comparison
        :param ref_plan: The reference plan to be used for the optimization process. If None, the first plan in the repository is used.
        :param gt_batch_size: The batch size for the ground truth data
        :param s0: The initial substitution plan to be used for the optimization process. If None, a random plan is selected.
        """
        self.repository = model_repository
        self.gt_data = gt_data
        self.work_dir = Path(work_dir)

        # Elements to be used during the optimization process
        self._best_substitution_plan = None
        self._simulator = Simulator.Simulator()
        self.substitution_plans = list(self.repository.model_repository.keys())
        self._gbms = list(self.repository.model_repository.values())
        self._risk_metric = risk_metric
        self._ref_plan = ref_plan if ref_plan is not None else self.substitution_plans[0]
        self._w1 = 0.5
        self._w2 = 0.5
        self._lambda2 = 1.
        self._lambda2 = 1.
        self._const_current_plan = None

        # If ref data is passed, save it according to the ref plan
        if ref_data is not None:
            save_to = self.work_dir / "simulations" / f"S_{self._ref_plan}.pkl"
            SimulationDataset.save(ref_data, save_to)

    def construct(self,
                  method="exhaustive_numerical",
                  w1=1.,
                  w2=1.,
                  lambda_1=0.5,
                  lambda_2=0.5,
                  save_in=None,
                  **kwargs,
                  ):
        """
        Constructs the best GreyBox model using greedy optimization.
        :return: (best_plan, performance_history)
        """
        methods = {"pseudo_random_heuristic": self._pseudo_random_heuristic,
                   "exhaustive_numerical": self._exhaustive_optimizer,
                   }
        try:
            optimizer = methods[method]
        except KeyError:
            raise ValueError("Method not found. Available methods are: {}".format(methods.keys()))

        result, info = optimizer(**kwargs)

        # Save the results
        if save_in is not None:
            d = {"method": method,
                 "result": result,
                 "info": info,
                 }
            path = Path(save_in) / f"{method}_results.pkl"
            with open(path, "wb") as f:
                pickle.dump(d, f)

            print(f"Saved results in: {path}")

        return result, info

    def get_best_model(self):
        """
        Returns the best model based on the performance criteria.
        """
        return

    def get_history(self):
        """
        Returns historical performance metrics.
        """
        pass

    def _exhaustive_optimizer(self,
                              n_scenarios=None,
                              parallel_plans=True,
                              parallel_scenarios=True,
                              plans=None,
                              ):
        """
        Screens all the substitution plans and selects the one that maximizes the VoI.

        :param n_scenarios: Number of scenarios to simulate. If None, all scenarios are used.
        :param parallel_plans: If True, simulates all plans in parallel.
        :param parallel_scenarios: If True, simulates all scenarios in parallel.
        :param plans: List of substitution plans to simulate. If None, all plans are used.
        :return: The best substitution plan and the performance metrics.
        """
        # If a number of scenarios is passed, extract them from the GT data
        scenarios = self.gt_data.extract(n_scenarios) if n_scenarios is not None else self.gt_data.scenarios

        # If not plans are passed, use all substitution plans
        if plans is None:
            plans = self.substitution_plans

        # The reference plant
        ref_plant = self.repository.get_model(self._ref_plan)

        # Prepare some info
        info = {"method": "exhaustive_numerical",
                "plans": plans,
                "number_of_scenarios": len(scenarios),
                "performance": {},
                }

        # Proceed to simulate the scenarios
        if parallel_plans:
            # Use parallel simulation
            self.parallel_simulate_plans(scenarios, plans, parallel=parallel_scenarios)

        else:
            # Use sequential simulation
            self.sequential_simulate_plans(scenarios, plans, parallel=parallel_scenarios)

        # Now, we need to compute the performance metrics
        info = {}
        # for plan in plans:
        #     # Get the reference simulation data
        #     sim_data_list = plan_simulations.scenarios
        #     scenario_ids = [s["id"] for s in sim_data_list]
        #
        #     # Get from the ref data the scenarios with the same id
        #     ref_sim_data_list = [self.gt_data.get_scenario_by_id(scenario_id) for scenario_id in scenario_ids]
        #
        #     # Compute the performance metrics
        #     l1 = computational_load(plan_simulations.scenarios)
        #     l2 = lack_of_fit(ref_sim_data_list,
        #                      plan_simulations.scenarios,
        #                      self._risk_metric,
        #                      ref_plant,
        #                      current_gbm
        #                      )
        #     voi = self.repository.voi(loss_fun=self.repository.plan_loss_variance_penalized,
        #                               w1=self._w1, w2=self._w2, w3=0.5,
        #                               plan=plan,
        #                               )
        #
        #     # Store the performance metrics
        #     info["performance"][plan] = {
        #         "computational_load": l1,
        #         "lack_of_fit": l2,
        #         "voi": voi,
        #     }

        # Find the model that maximizes voi
        best_s = max(info, key=lambda x: info["performance"][x]["voi"])

        return best_s, info

    def _pseudo_random_heuristic(self,
                                 w1=1.,
                                 w2=1.,
                                 lambda_variance=0.5,
                                 n_prev_plans=3,
                                 save_in=None,
                                 **kwargs,
                                 ):
        """
        Performs a greedy optimization process for selecting the best GreyBox substitution plan.
        """
        perf = {"method": "greedy_voi"}

        # Compute priors
        self.repository.prior_values(simulation_data_list=self.gt_data.scenarios,
                                     risk_metric=self._risk_metric,
                                     num_scenarios=self._prior_num_scenarios,
                                     batch_size=self._prior_batch_size,
                                     )

        self._prior_num_scenarios = 50
        self._prior_batch_size = 10

        # Get the batches
        batches = self.gt_data.get_batches(self.gt_batch_size)

        # History of plans
        plan_history = []

        # Select a first substitution plan, the heuristic knows how to do it
        current_s = self._next_substitution_plan_heuristic(plan_history, n_prev_plans) if self._s0 is None else self._s0

        # Iterate through all batches
        for gt_batch in tqdm.tqdm(batches):
            # Simulate the batch
            gbm_batch = self._opt_simulate_batch(current_s, gt_batch, self.repository)

            # Update estimators
            self.repository.update_performance(plan=current_s,
                                               sim_data_list=gbm_batch,
                                               gt_data_list=gt_batch,
                                               risk_metric=self._risk_metric,
                                               )
            # Add to history
            plan_history.append(current_s)

            # Select the next plan to simulate
            current_s = self._next_substitution_plan_heuristic(plan_history, n_prev_plans)

            # Save history
            if save_in is not None:
                _history = self._get_history()
                _history["plan_history"] = plan_history
                path = Path(save_in) / f"_temp_history_greedy_voi.pkl"
                with open(path, "wb") as f:
                    pickle.dump(_history, f)

        # Select the final best model
        voi = self.repository.voi(loss_fun=self.repository.plan_loss,
                                  w1=self._w1, w2=self._w2)
        history = self._get_history()

        best_plan = max(voi, key=lambda plan: voi[plan]['voi'])
        perf["best_plan"] = best_plan
        perf["history"] = history
        perf["last_voi"] = voi

        # Save
        if save_in is not None:
            path = Path(save_in) / f"greedy_voi_results.pkl"
            with open(path, "wb") as f:
                pickle.dump(perf, f)
            print(f"Saved results in: {path}")

        return best_plan, perf

    def _next_substitution_plan_heuristic(self, plan_history, n_prev_plans):
        """
        The heuristic uses VoI to select the next substitution plant in the iteration
        """
        # Check start
        if self._s0 is None:
            valid_s = [S for S in self.substitution_plans if any(S)]
            return random.choice(valid_s)

        # Heuristic
        max_l1 = self.repository.model_performance[self._ref_plan]["computational_load"].mu_history[0]
        self._w1 = 1 / max_l1  #TODO fix this crap
        self._w2 = 1.
        last_plans = plan_history[-n_prev_plans:] if len(plan_history) > n_prev_plans else plan_history
        voi = self.repository.voi(loss_fun=self.repository.plan_loss_variance_penalized,
                                  w1=self._w1, w2=1., w3=0.5, )

        # Remove recently used plans from VoI selection
        eligible_voi = {plan: value for plan, value in voi.items() if plan not in last_plans}

        # If no eligible plans exist, select randomly from all substitution plans
        if not eligible_voi:
            return random.choice(self.substitution_plans)

        # Convert VoI values into probabilities for selection
        voi_values = np.array([x['voi'] for x in eligible_voi.values()])

        # Apply softmax to convert VoI into probabilities (ensures higher VoI means higher selection probability)
        exp_voi = np.exp(voi_values - np.max(voi_values))
        probabilities = exp_voi / np.sum(exp_voi)

        # Randomly sample a plan using the computed probabilities
        selected_plan = random.choices(population=list(eligible_voi.keys()), weights=probabilities, k=1)[0]

        return selected_plan

    @staticmethod
    def _simulate_plan(plan, scenarios, repository, parallel=True, pbar_offset=0):
        """Simulates all scenarios for a given plan (must be picklable for multiprocessing)."""
        gbm_outputs = GreyBoxModelConstructor._opt_simulate_batch(plan, scenarios, repository, parallel=parallel,
                                                                  pbar_offset=pbar_offset)

        # Create a SimulationDataset from the simulation data
        scenario_ids = [s["id"] for s in gbm_outputs]
        sim_dataset = SimulationDataset.SimulationDataset(gbm_outputs, scenario_ids)

        # Process the simulation data
        sim_dataset = sim_dataset.process_scenarios(SimulationDataset.default_process_scenario)

        return sim_dataset

    @staticmethod
    def _opt_simulate_batch(substitution_plan, scenarios, repository, parallel=True, pbar_offset=0):
        """
        A static version of `_opt_simulate_batch()` that can be called in parallel.
        It retrieves the model and runs the simulation.
        """
        gbm = repository.get_model(substitution_plan)  # Use the repository instance
        gbm.stochastic = False  # Disable stochasticity always

        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            # with ThreadPoolExecutor() as executor:
            #     futures = []
            #     for i, scenario in enumerate(scenarios):
            #         gbm_copy = copy.deepcopy(gbm)
            #         futures.append(
            #             executor.submit(GreyBoxModelConstructor._simulate_scenario, scenario, gbm_copy,
            #                             substitution_plan,
            #                             pbar_offset + i + 1))
            #     results = [future.result() for i, future in enumerate(tqdm.tqdm(futures, desc="Simulating scenarios",
            #                                                                     position=pbar_offset + i))]

            with ThreadPoolExecutor() as executor:
                futures = []
                for i, scenario in enumerate(scenarios):
                    gbm_copy = copy.deepcopy(gbm)
                    futures.append(
                        executor.submit(GreyBoxModelConstructor._simulate_scenario, scenario, gbm_copy, substitution_plan,
                                        pbar_offset+i+1))
                results = [future.result() for future in tqdm.tqdm(futures, desc="Simulating scenarios", position=pbar_offset)]
        else:
            results = []
            for i, scenario in enumerate(tqdm.tqdm(scenarios, desc="Simulating scenarios", position=pbar_offset)):
                result = GreyBoxModelConstructor._simulate_scenario(scenario, gbm, substitution_plan, pbar_offset+i+1)
                results.append(result)

        return results

    @staticmethod
    def _simulate_scenario(scenario, gbm, substitution_plan, position):
        """
        Simulates a single scenario using the given Grey-Box Model (GBM).
        """
        try:
            # Time
            t = scenario["time"]
            mission_time = scenario["mission_time"]
            dt = t[1] - t[0]

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

            params = Simulator.SimulationParameters(
                initial_time=t0,
                mission_time=mission_time,
                time_step=dt,
                initial_state=x0,
                external_stimuli=e,
                forced_states=x,
            )
            simulator = Simulator.Simulator()
            simulator.params = params
            simulator.plant = gbm
            sim_data = simulator.simulate(pbar_post=f"S={substitution_plan} | Scenario: {scenario['id']}",
                                          pbar_position=position)

            # Add the scenario ID to the simulation data
            sim_data["id"] = scenario["id"]
            return sim_data

        except Exception as e:
            tqdm.tqdm.write(f"Failed ({e}): S={substitution_plan} | Scenario: {scenario['id']}")
            # print(f"Error simulating scenario {scenario['id']} with plan {substitution_plan}: {e}")
            return None

    def _get_history(self):
        """
        Returns the history of the constructor.
        """
        history_l1 = {p: self.repository.model_performance[p]["computational_load"].get_history()
                      for p in self.substitution_plans}
        history_l2 = {p: self.repository.model_performance[p]["lack_of_fit"].get_history()
                      for p in self.substitution_plans}

        return {"history_l1": history_l1, "history_l2": history_l2}

    def parallel_simulate_plans(self,
                                scenarios,
                                plans=None,
                                parallel=True,
                                ):
        """
        Parallelizes the simulation for each plan.

        :param scenarios: List of scenarios to simulate.
        :param plans: List of substitution plans to simulate. If None, all plans are used.
        :return: Dictionary with plans as keys and their simulation results as values.
        """
        if plans is None:
            plans = self.substitution_plans

        # Function to simulate a single plan
        def simulate_plan(plan):
            simulation_path = self.work_dir / "simulations" / Path(f"S_{plan}.pkl")
            try:
                # Check if the simulation outcomes are already computed
                with open(simulation_path, "rb") as f:
                    plan_simulations = pickle.load(f)
            except FileNotFoundError:
                # If not, simulate the scenarios
                print(f"Simulating plan: {plan}")
                plan_simulations = []
                for scenario in tqdm.tqdm(scenarios, desc=f"Simulating {plan}"):
                    result = self._simulate_plan(plan, scenario, self.repository, parallel=parallel)
                    plan_simulations.append(result)
                # Save the results to a file
                with open(simulation_path, "wb") as f:
                    pickle.dump(plan_simulations, f)
            return plan, plan_simulations

        # Use multiprocessing to parallelize the simulation
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(simulate_plan, plans)

        # Convert results to a dictionary
        simulation_results = {plan: sim_data for plan, sim_data in results}
        return simulation_results

    def sequential_simulate_plans(self, scenarios, plans, parallel=False):
        """
        Sequentially simulates the scenarios for each plan.

        :param scenarios: List of scenarios to simulate.
        :param plans: List of substitution plans to simulate.
        :return: Dictionary with plans as keys and their simulation results as values.
        """
        plan_simulations_dict = {}
        pbar_offset = 0
        for plan in plans:
            simulation_path = self.work_dir / "simulations" / Path(f"S_{plan}.pkl")
            try:
                # Check if the simulation outcomes are already computed
                with open(simulation_path, "rb") as f:
                    plan_simulations = pickle.load(f)

                print(f"[LOADED] Data for plan {plan} from {simulation_path}")

            except FileNotFoundError:
                # If not, simulate the scenarios
                print(f"[NOT LOADED] Data for plan {plan}...")
                print(f"    ->  Simulating plan: {plan}")
                plan_simulations = self._simulate_plan(plan, scenarios, self.repository, parallel=parallel,
                                                       pbar_offset=0)
                pbar_offset += len(scenarios)

                # Save the results to a file
                print(f"    -> Saving simulation data for plan {plan} to {simulation_path}")
                with open(simulation_path, "wb") as f:
                    pickle.dump(plan_simulations, f)

                print("    -> Done!")

            plan_simulations_dict[plan] = plan_simulations
        return plan_simulations_dict

"""
Utility functions
"""


def load(path):
    with open(path, "rb") as f:
        gc = pickle.load(f)

    return gc


def save(gc, path):
    with open(path, "wb") as f:
        pickle.dump(gc, f)


"""
Functions for computational load and lack of fit
"""


def computational_load(sim_data_list):
    """
    Computes the computational load of each simulation data dictionary in the list.

    :param sim_data_list: List of dictionaries with simulation data.
    :return: The computational load of the simulation.
    """

    def slope_fit(t_sim, t_exec):
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

    results = []
    info = []
    for sim_data in sim_data_list:
        # Get necessary data
        t_sim = np.array(sim_data["time"])
        t_exec = np.array(sim_data["execution_time_array"])
        t_exec = t_exec - t_exec[0]
        m = slope_fit(t_sim, t_exec)
        results.append(m)
        info.append({"t_sim": t_sim, "t_exec": t_exec, "slope": m})

    mean_result = np.mean(results)
    variance_result = np.var(results)
    info_dict = {
        "mean": mean_result,
        "variance": variance_result,
        "details": info
    }

    return (mean_result, variance_result), info_dict


def lack_of_fit(ref_sim_data_list,
                sim_data_list,
                risk_metric,
                plant_ref,
                plant_gbm,
                ):
    """
    Computes the lack of fit between two simulation data dictionaries.

    :param ref_sim_data_list: Reference simulation data dictionary.
    :param sim_data_list: Simulation data dictionary.
    :param risk_metric: Risk metric to use for lack of fit.
    :param risk_aggregation: Risk aggregation function.
    :param plant_ref: Reference plant.
    :param plant_gbm: Grey-Box Model.
    :return: The lack of fit between the two simulations.
    """

    def ks_statistic(data_ref, data, n_bins=20):
        """
        Compute the Kolmogorov-Smirnov statistic between two empirical cumulative distribution functions.
        :param data: the data to compare
        :param data_ref: the reference data
        :param n_bins: the number of bins to use
        """
        # If both are the same, return 0
        if np.array_equal(data_ref, data):
            return 0, {"equal_data": True}

        # Generate shared bin edges for both datasets
        min_x = min(data.min(), data_ref.min())
        max_x = max(data.max(), data_ref.max())
        bins = np.linspace(min_x, max_x, n_bins + 1)

        # Compute empirical PDFs
        epdf_ref, _ = np.histogram(data_ref, bins=bins, density=True)
        epdf, _ = np.histogram(data, bins=bins, density=True)

        # Compute empirical CDFs
        ecdf_ref = empirical_cdf(bins, epdf_ref)
        ecdf = empirical_cdf(bins, epdf)

        # Compute KS statistic
        abs_diff = np.abs(ecdf - ecdf_ref)
        ks_value = np.max(abs_diff)
        max_diff_idx = np.argmax(abs_diff)
        ks_loc = bins[max_diff_idx]

        # Store information
        info = {"ks_idx": max_diff_idx,
                "ks_value": ks_value,
                "ks_bin_loc": ks_loc,
                "bins": bins,
                "epdf": epdf,
                "epdf_ref": epdf_ref,
                "ecdf": ecdf,
                "ecdf_ref": ecdf_ref,
                "abs_diff": abs_diff,
                }

        return ks_value, info

    def empirical_cdf(bin_edges, epdf):
        """
        Compute the empirical cumulative distribution function (ECDF) from an empirical PDF.

        :param bin_edges: Bin edges from np.histogram (length n_bins + 1)
        :param epdf: Empirical probability density function (length n_bins)
        :return: ECDF values at each bin edge
        """
        # Compute cumulative sum of the PDF to get the ECDF
        cdf_values = np.cumsum(epdf * np.diff(bin_edges))

        # Normalize so that ECDF ranges from 0 to 1
        cdf_values /= cdf_values[-1]  # Divide by last value to make it 1

        # Prepend a 0 at the start to ensure ECDF starts at 0
        cdf_values = np.concatenate(([0], cdf_values))

        return cdf_values

    def aggregate(t: np.ndarray, r: np.ndarray) -> float:
        """
        Aggregates the risk metric r
        :param t: time points
        :param r: risk metric values
        :return: aggregated risk metric
        """
        # from scipy.integrate import simpson
        # return simpson(r, t)
        return np.sum(r) / len(r)  # Simple average

    # Compute the metric from each simulation
    metric_ref = [risk_metric(ref_scenario, plant_ref) for ref_scenario in ref_sim_data_list]
    metric = [risk_metric(scenario, plant_gbm) for scenario in sim_data_list]

    # Aggregate the metric (one per simulation)
    agg_metric_ref = [aggregate(x["time"], m) for x, m in zip(ref_sim_data_list, metric_ref)]
    agg_metric = [aggregate(x["time"], m) for x, m in zip(sim_data_list, metric)]

    # Turn into np array for KS computation
    agg_metric_ref = np.array(agg_metric_ref)
    agg_metric = np.array(agg_metric)

    # Compute KS statistic for the aggregated metric
    ks, info_dict = ks_statistic(agg_metric_ref, agg_metric)

    # Store the results
    mean_result = ks
    variance_result = ks * 0.05 + 1e-6

    return (mean_result, variance_result), info_dict


def aggregated_loss(mu_l1, mu_l2, sigma_l1, sigma_l2, omega1, omega2, lambdal1, lambdal2, alpha=1):
    l = (omega1 * lambdal1 * mu_l1) + (omega2 * lambdal2 * mu_l2) + alpha * np.sqrt(omega1 ** 2 * lambdal1 ** 2 * sigma_l1 ** 2 + omega2 ** 2 * lambdal2 ** 2 * sigma_l2 ** 2)
    return l



def lack_of_fit_OLD(ref_sim_data_list,
                sim_data_list,
                risk_metric,
                plant_ref,
                plant_gbm,
                ):
    """
    Computes the lack of fit between two simulation data dictionaries.

    :param ref_sim_data_list: Reference simulation data dictionary.
    :param sim_data_list: Simulation data dictionary.
    :param risk_metric: Risk metric to use for lack of fit.
    :param risk_aggregation: Risk aggregation function.
    :param plant_ref: Reference plant.
    :param plant_gbm: Grey-Box Model.
    :return: The lack of fit between the two simulations.
    """

    def ks_statistic(data_ref, data, n_bins=10):
        """
        Compute the Kolmogorov-Smirnov statistic between two empirical cumulative distribution functions.
        :param data: the data to compare
        :param data_ref: the reference data
        :param n_bins: the number of bins to use
        """
        # If both are the same, return 0
        if np.array_equal(data_ref, data):
            return 0, {"equal_data": True}

        # Generate shared bin edges for both datasets
        min_x = min(data.min(), data_ref.min())
        max_x = max(data.max(), data_ref.max())
        bins = np.linspace(min_x, max_x, n_bins + 1)

        # Compute empirical PDFs
        epdf_ref, _ = np.histogram(data_ref, bins=bins, density=True)
        epdf, _ = np.histogram(data, bins=bins, density=True)

        # Compute empirical CDFs
        ecdf_ref = empirical_cdf(bins, epdf_ref)
        ecdf = empirical_cdf(bins, epdf)

        # Compute KS statistic
        abs_diff = np.abs(ecdf - ecdf_ref)
        ks_value = np.max(abs_diff)
        max_diff_idx = np.argmax(abs_diff)
        ks_loc = bins[max_diff_idx]

        # Store information
        info = {"ks_idx": max_diff_idx,
                "ks_value": ks_value,
                "ks_bin_loc": ks_loc,
                "bins": bins,
                "epdf": epdf,
                "epdf_ref": epdf_ref,
                "ecdf": ecdf,
                "ecdf_ref": ecdf_ref,
                "abs_diff": abs_diff,
                }

        return ks_value, info

    def empirical_cdf(bin_edges, epdf):
        """
        Compute the empirical cumulative distribution function (ECDF) from an empirical PDF.

        :param bin_edges: Bin edges from np.histogram (length n_bins + 1)
        :param epdf: Empirical probability density function (length n_bins)
        :return: ECDF values at each bin edge
        """
        # Compute cumulative sum of the PDF to get the ECDF
        cdf_values = np.cumsum(epdf * np.diff(bin_edges))

        # Normalize so that ECDF ranges from 0 to 1
        cdf_values /= cdf_values[-1]  # Divide by last value to make it 1

        # Prepend a 0 at the start to ensure ECDF starts at 0
        cdf_values = np.concatenate(([0], cdf_values))

        return cdf_values

    def aggregate(t: np.ndarray, r: np.ndarray) -> float:
        """
        Aggregates the risk metric r using integration.
        :param t: time points
        :param r: risk metric values
        :return: aggregated risk metric
        """
        from scipy.integrate import simpson
        return simpson(r, t)

    def ensure_full_coverage_indices(N, Ns=3):
        """
        Ensures that in N random samplings of Ns elements, all elements are selected at least once.

        :param N: Total number of elements.
        :param Ns: Number of elements to sample in each iteration.
        :return: List of sampled subsets (indices of the original list).
        """
        if Ns >= N:
            raise ValueError("Ns must be smaller than the number of elements N.")

        sampled_indices = []
        remaining_indices = set(range(N))  # Track indices that haven't been included yet

        for _ in range(N):
            if len(remaining_indices) >= Ns:
                # Preferentially sample from indices that haven't been included yet
                sampled = np.random.choice(list(remaining_indices), size=Ns, replace=False)
            else:
                # Sample randomly while ensuring we include remaining unseen indices
                needed = list(remaining_indices)
                additional = np.random.choice(range(N), size=Ns - len(needed), replace=False)
                sampled = np.array(needed + list(additional))

            sampled_indices.append(sampled)
            remaining_indices -= set(sampled)  # Remove newly covered indices

            # Reset remaining elements if all have been covered
            if not remaining_indices:
                remaining_indices = set(range(N))

        return sampled_indices

    # Get the combinations
    comb_idx = ensure_full_coverage_indices(len(ref_sim_data_list), Ns=3)

    # Storage
    ks_list = []
    info_list = []
    for idx in comb_idx:
        # Use the index to get the target scenarios
        s_ref = [ref_sim_data_list[i] for i in idx]
        s = [sim_data_list[i] for i in idx]

        # Compute the metric from each simulation
        metric_ref = [risk_metric(ref_sim_data, plant_ref) for ref_sim_data in s_ref]
        metric = [risk_metric(sim_data, plant_gbm) for sim_data in s]

        # Aggregate the metric (one per simulation)
        agg_metric_ref = [aggregate(x["time"], m) for x, m in zip(s_ref, metric_ref)]
        agg_metric = [aggregate(x["time"], m) for x, m in zip(s, metric)]

        # Turn into np array for KS computation
        agg_metric_ref = np.array(agg_metric_ref)
        agg_metric = np.array(agg_metric)

        # Compute KS statistic for the aggregated metric
        ks, info = ks_statistic(agg_metric_ref, agg_metric)
        ks_list.append(ks)
        info_list.append(info)

    return ks_list, info_list