"""
Iterative method to guide GBM construction using VoI, coupled with some optimization algorithm.

Author: Juan-Pablo Futalef
"""
#%%
import copy
import random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import tqdm
from typing import List, Tuple

from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository
from greyboxmodels.construction.GroundTruth import GroundTruthDataset
from greyboxmodels.modelbuild import Input
from greyboxmodels.simulation import Simulator


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: GroundTruthDataset,
                 gt_batch_size: int = 5,
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
        self._best_substitution_plan = None
        self._simulator = Simulator.Simulator()
        self._const_current_plan = None
        self.substitution_plans = list(self.repository.model_repository.keys())

    def construct(self,
                  ):
        """
        Constructs the best GreyBox model using greedy optimization.
        :return: (best_plan, performance_history)
        """
        best_plan, history = self._greedy_optimizer()
        self._best_substitution_plan = best_plan  # Store the best plan
        return best_plan, history

    def get_best_model(self):
        """
        Returns the best model based on the performance criteria.
        """
        return

    def get_history(self):
        """
        Returns historical performance metrics.
        """
        history = {"computational_burden": {}, "fidelity": {}}

        for plan in self.repository.keys():
            history["computational_burden"][plan] = self.computational_burden_estimators[plan].get_history()
            history["fidelity"][plan] = self.fidelity_estimators[plan].get_history()

        return history

    def _greedy_optimizer(self,
                          ):
        """
        Performs a greedy optimization process for selecting the best GreyBox substitution plan.
        """
        # Get the batches
        batches = self.gt_data.get_batches(self.gt_batch_size)

        # Select a first substitution plan, the heuristic knows how to do it
        current_s = (0, 1, 1)
        # current_s = self._next_substitution_plan_heuristic()

        # Iterate through all batches
        for batch in tqdm.tqdm(batches):
            # Simulate the batch
            simulated_data_batch = self._opt_simulate_batch(current_s, batch)

            # Update estimators
            self.repository.update_performance(current_s, simulated_data_batch)  # Every update stores history

            # Select the next plan to simulate
            current_s = self._next_substitution_plan_heuristic()

        # Select the final best model
        best_plan = self.get_best_model()
        history = self.get_history()
        return best_plan, history

    def _next_substitution_plan_heuristic(self):
        """
        The heuristic uses VoI to select the next substitution plant in the iteration
        """
        # Check start
        if self._best_substitution_plan is None:
            valid_S = [S for S in self.substitution_plans if any(S)]
            S = random.choice(valid_S)
            return S

        # From the repository, use the meand and std to compute the relative error
        # Then, use the relative error to compute the VoI
        # Finally, select the plan with the highest VoI
        pass

    def _opt_simulate_batch(self,
                            substitution_plan,
                            batch,
                            ):
        """
        Simulates a batch of data using the given substitution plan.
        """
        # Get the model
        gbm = self.repository.get_model(substitution_plan)
        gbm.stochastic = False  # Disable stochasticity always

        # Simulate
        results = []
        for scenario in batch:
            params = Simulator.SimulationParameters(initial_time=scenario["initial_time"],
                                                    mission_time=scenario["mission_time"],
                                                    time_step=scenario["time_step"],
                                                    initial_state=scenario["initial_state"],
                                                    external_stimuli=scenario["external_stimuli"],
                                                    forced_states=scenario["forced_states"],
                                                    )
            self._simulator.params = params
            self._simulator.plant = gbm
            sim_data = self._simulator.simulate()
            results.append(sim_data)

        return results

    def _opt_update(self, substitution_plan, simulated_data_batch):
        """
        Updates the Bayesian estimators with new performance data.
        """
        # Measure computational burden
        z_comp_burden = simulated_data_batch["execution_time"]

        # Measure fidelity
        z_fidelity = non_zero_error(simulated_data_batch["state"], simulated_data_batch["state_variables"])

        # Update computational burden estimator
        self.computational_burden_estimators[substitution_plan].update(z_comp_burden)

        # Update fidelity estimator
        self.fidelity_estimators[substitution_plan].update(z_fidelity)

    def _opt_compute_voi(self):
        """
        Computes Value of Information (VoI) for all substitution plans.
        """
        subs_list = self.substitution_plan_list()
        Sref = subs_list[0]  # Reference plan (WBM)
        voi_values = {}
        for plan in subs_list[1:]:
            # Get those values for the reference plan (WBM)
            ref_mean_comp_burd, ref_std_comp_burd = self.computational_burden_estimators[Sref].get_distribution()
            ref_mean_fidelity, ref_std_fidelity = self.fidelity_estimators[Sref].get_distribution()

            # Get those values for the current plan
            mean_comp_burd, std_comp_burd = self.computational_burden_estimators[plan].get_distribution()
            mean_fidelity, std_fidelity = self.fidelity_estimators[plan].get_distribution()

            # Aggregate each
            Lref = ref_mean_comp_burd + ref_mean_fidelity
            L = mean_comp_burd + mean_fidelity

            # Compute VoI
            VoI_plan = Lref - L

            # Store
            voi_values[plan] = VoI_plan

        return voi_values


"""
Utility functions
"""


def greedy_optimizer(model_constructor, num_iterations=None):
    """
    Performs a greedy optimization process for selecting the best GreyBox substitution plan.

    Parameters:
    - model_constructor (GreyBoxModelConstructor): The object managing the model repository and estimators.
    - num_iterations (int, optional): The number of iterations (if None, will be based on batch size & total scenarios).

    Returns:
    - best_plan (tuple): The best substitution plan found.
    - performance_history (dict): Evolution of mean computational burden & fidelity per iteration.
    """
    # Get the batches
    batches = model_constructor.gt_data.get_batches(model_constructor.gt_batch_size)

    # Get substitution plans
    s_list = model_constructor.substitution_plan_list()

    # Select randomly the first substitution plan. It should have a single one in some position
    while current_s := np.random.choice(s_list):
        if 1 in current_s:
            break

    # Iterate
    for batch in tqdm.tqdm(batches):
        # Simulate the batch
        simulated_data_batch = model_constructor._opt_simulate_batch(current_s, batch)

        # Update estimators
        model_constructor._opt_update(current_s, simulated_data_batch)  # Every update stores history

        # Use updated estimators to compute VoI for all substitution plans
        voi_values = model_constructor._opt_compute_voi(s_list)

        # Select the next plan
        current_s = next_substitution_plan_heuristic(voi_values)

    # Select the final best model
    best_plan = model_constructor.get_best_model()
    history = model_constructor.get_history()
    return best_plan, history


def next_substitution_plan_heuristic():
    pass
