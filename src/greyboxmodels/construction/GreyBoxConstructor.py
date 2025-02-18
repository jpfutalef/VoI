"""
Iterative method to guide GBM construction using VoI, coupled with some optimization algorithm.

Author: Juan-Pablo Futalef
"""
import copy
import random

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool
import dill as pickle
import tqdm
import multiprocessing as mp
from pathlib import Path

from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository, computational_load, lack_of_fit
from greyboxmodels.construction.GroundTruth import GroundTruthDataset
from greyboxmodels.simulation import Simulator
from greyboxmodels.modelbuild import Input


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: GroundTruthDataset,
                 risk_metric: callable,
                 gt_batch_size: int = 5,
                 s0=None,
                 ref_plan=None,
                 ):
        """
        This constructor enables the iterative selection of the best model from a repository of models
        and a dataset of ground truth data. The selection is guided by the Value of Information (VoI) metric.

        :param model_repository: A repository of grey-box models
        :param gt_data: Ground truth data
        :param params: Constructor parameters
        """
        self.repository = model_repository
        self.gt_data = copy.deepcopy(gt_data)
        self.gt_batch_size = gt_batch_size

        # Elements to be used during the optimization process
        self._s0 = s0
        self._best_substitution_plan = None
        self._simulator = Simulator.Simulator()
        self._const_current_plan = None
        self.substitution_plans = list(self.repository.model_repository.keys())
        self._gbms = list(self.repository.model_repository.values())
        self._risk_metric = risk_metric
        self._ref_plan = ref_plan if ref_plan is not None else self.substitution_plans[0]
        self._prior_num_scenarios = 50
        self._prior_batch_size = 10
        self._w1 = 1.
        self._w2 = 1.

    def construct(self,
                  method="greedy_voi",
                  **kwargs,
                  ):
        """
        Constructs the best GreyBox model using greedy optimization.
        :return: (best_plan, performance_history)
        """
        methods = {"greedy_voi": self._greedy_optimizer,
                   "exhaustive_numerical": self._exhaustive_optimizer,
                   }
        try:
            optimizer = methods[method]
        except KeyError:
            raise ValueError("Method not found. Available methods are: {}".format(methods.keys()))

        return optimizer(**kwargs)

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
                              save_in=None,
                              w1=1.,
                              w2=1.,
                              lambda_variance=0.5,
                              n_scenarios=5,
                              parallel=True,
                              plans=None,
                              ):
        """
        Computes the performances of all substitution plans and selects the best one.
        If `parallel=True`, executes in parallel using multiple processes.
        """
        perf = {"method": "exhaustive_numerical"}

        # Extract a shuffled subset of scenarios
        scenarios = self.gt_data.extract(n_scenarios)

        # If a plan is passed, use it
        if plans is None:
            plans = self.substitution_plans

        if parallel:
            with ProcessPool() as pool:
                results = pool.map(GreyBoxModelConstructor._simulate_plan,
                                   plans,
                                   [scenarios] * len(plans),
                                   [self.repository] * len(plans),
                                   )

            for plan, result in results:
                perf[plan] = result
                # Save if save_in is passed
                if save_in is not None:
                    path = save_in / f"S_{plan}.pkl"
                    with open(path, "wb") as f:
                        pickle.dump(result, f)
                    print(f"Saved outcomes in: {path}")

        else:
            for plan in plans:
                # Check if file is already computed
                if save_in is not None:
                    path = save_in / f"S_{plan}.pkl"
                    if path.exists():
                        with open(path, "rb") as f:
                            result = pickle.load(f)
                        print(f"Loaded outcomes from: {path}")
                        perf[plan] = result
                        continue

                # Otherwise, compute the outcomes
                result = self._simulate_plan(plan, scenarios, self.repository)
                perf[plan] = result

                # Update estimators
                self.repository.update_performance(plan,
                                                   result,
                                                   scenarios,
                                                   risk_metric=self._risk_metric,
                                                   )

                # Save if save_in is passed
                if save_in is not None:
                    path = save_in / f"S_{plan}.pkl"
                    with open(path, "wb") as f:
                        pickle.dump(result, f)
                    print(f"Saved outcomes in: {path}")

        # Obtain the best model
        perf["voi"] = self.repository.voi(w1=1., w2=1.)

        # Find the model that maximizes voi
        best_model = max(perf, key=lambda x: perf["voi"][x])

        return best_model, perf

    def _greedy_optimizer(self,
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
        self._w1 = 1 / max_l1   #TODO fix this crap
        self._w2 = 1.
        last_plans = plan_history[-n_prev_plans:] if len(plan_history) > n_prev_plans else plan_history
        voi = self.repository.voi(loss_fun=self.repository.plan_loss_variance_penalized,
                                  w1=self._w1, w2=1., w3=0.5,)

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
    def _simulate_plan(plan, scenarios, repository):
        """Simulates all scenarios for a given plan (must be picklable for multiprocessing)."""
        gbm_outputs = GreyBoxModelConstructor._opt_simulate_batch(plan, scenarios, repository)
        return gbm_outputs

    @staticmethod
    def _opt_simulate_batch(substitution_plan, scenarios, repository):
        """
        A static version of `_opt_simulate_batch()` that can be called in parallel.
        It retrieves the model and runs the simulation.
        """
        gbm = repository.get_model(substitution_plan)  # Use the repository instance
        gbm.stochastic = False  # Disable stochasticity always

        results = []
        for scenario in scenarios:
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
            sim_data = simulator.simulate(pbar_post=f"S={substitution_plan}")
            results.append(sim_data)

        return results

    def _get_history(self):
        """
        Returns the history of the constructor.
        """
        history_l1 = {p: self.repository.model_performance[p]["computational_load"].get_history()
                      for p in self.substitution_plans}
        history_l2 = {p: self.repository.model_performance[p]["lack_of_fit"].get_history()
                      for p in self.substitution_plans}

        return {"history_l1": history_l1, "history_l2": history_l2}


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
