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
        self._risk_metric = risk_metric

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

    def _exhaustive_optimizer(self, w1=1., w2=1., n_scenarios=5, parallel=True):
        """
        Computes the performances of all substitution plans and selects the best one.
        If `parallel=True`, executes in parallel using multiple processes.
        """
        perf = {"method": "exhaustive_numerical"}

        # Extract a shuffled subset of scenarios
        scenarios = self.gt_data.extract(n_scenarios)

        if parallel:
            manager = mp.Manager()
            models_data = manager.dict(self.repository.model_repository)
            with ProcessPool() as pool:
                results = pool.map(GreyBoxModelConstructor._simulate_plan,
                                   self.substitution_plans,
                                   [scenarios] * len(self.substitution_plans),
                                   [models_data] * len(self.substitution_plans),
                                   )

            for plan, result in results:
                perf[plan] = result
        else:
            for plan in self.substitution_plans:
                plan, result = self._simulate_plan(plan, scenarios, self.repository)
                perf[plan] = result

        # Obtain the best model
        sums = {plan: w1 * p["computational_burden"] + w2 * p["fidelity"] for plan, p in perf.items()}
        best_model = min(sums, key=sums.get)

        return best_model, perf

    def _greedy_optimizer(self, w1=1., w2=1., **kwargs):
        """
        Performs a greedy optimization process for selecting the best GreyBox substitution plan.
        """
        perf = {"method": "greedy_voi"}

        # Get the batches
        batches = self.gt_data.get_batches(self.gt_batch_size)

        # Select a first substitution plan, the heuristic knows how to do it
        current_s = self._next_substitution_plan_heuristic() if self._s0 is None else self._s0

        # Iterate through all batches
        for batch in tqdm.tqdm(batches):
            # Simulate the batch
            simulated_data_batch = self._opt_simulate_batch(current_s, batch, self.repository)

            # Update estimators
            self.repository.update_performance(current_s,
                                               simulated_data_batch,
                                               batch,
                                               risk_metric=self._risk_metric,
                                               )  # Every update stores history

            # Select the next plan to simulate
            current_s = self._next_substitution_plan_heuristic()

        # Select the final best model
        best_plan = self.get_best_model()
        history = self.get_history()

        perf["best_plan"] = best_plan
        perf["history"] = history

        return best_plan, perf

    def _next_substitution_plan_heuristic(self):
        """
        The heuristic uses VoI to select the next substitution plant in the iteration
        """
        # Check start
        if self._best_substitution_plan is None:
            valid_s = [S for S in self.substitution_plans if any(S)]
            return random.choice(valid_s)

        # From the repository, use the mean and std to compute the relative error
        # Then, use the relative error to compute the VoI
        # Finally, select the plan with the highest VoI
        pass

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
        # TODO create Input here
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
