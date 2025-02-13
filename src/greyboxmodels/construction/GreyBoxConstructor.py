"""
Iterative method to guide GBM construction using VoI, coupled with some optimization algorithm.

Author: Juan-Pablo Futalef
"""
#%%
import copy
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import tqdm

from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository
from greyboxmodels.construction.GroundTruth import GroundTruthDataset
from greyboxmodels.modelbuild import Input
from greyboxmodels.simulation import Simulator


class BayesianNormalEstimator:
    """
    A Bayesian estimator for tracking the mean and standard deviation of normally distributed variables.

    This class uses Bayesian updates to refine estimates of a normal distribution's parameters (mean & std deviation)
    based on new independent and identically distributed (IID) normal observations.

    Attributes:
    - mu (np.ndarray): Estimated mean(s) of the normal distribution.
    - sigma (np.ndarray): Estimated standard deviation(s).
    - mu_history (list): Stores evolution of mean estimates over updates.
    - sigma_history (list): Stores evolution of standard deviation estimates.
    """

    def __init__(self, mu_prior, sigma_prior):
        """
        Initializes the Bayesian normal estimator with prior mean and standard deviation.

        :param mu_prior: Prior mean (scalar or array for multiple variables).
        :param sigma_prior: Prior standard deviation (same shape as mu_prior).
        """
        self.mu = np.array(mu_prior, dtype=np.float64)  # Convert to NumPy array
        self.sigma = np.array(sigma_prior, dtype=np.float64)  # Convert to NumPy array

        # Track history for visualization
        self.mu_history = [self.mu.copy()]
        self.sigma_history = [self.sigma.copy()]

    def update(self, observations):
        """
        Performs a Bayesian update on the mean and standard deviation based on new observations.

        :param observations: A batch of normally distributed observations.
                             Should be a list of lists if tracking multiple variables.
        """
        observations = np.array(observations, dtype=np.float64)  # Convert input to NumPy array
        m = observations.shape[0]  # Number of new samples

        if m == 0:
            return  # No update if no new data

        # Compute sample mean and standard deviation from the new observations
        mu_Y = np.mean(observations, axis=0)  # Mean per variable
        sigma_Y = np.std(observations, axis=0, ddof=1)  # Std deviation per variable

        # Prior variance and sample variance
        sigma_prior_sq = self.sigma ** 2
        sigma_Y_sq = sigma_Y ** 2

        # Bayesian update formulas
        mu_new = (sigma_prior_sq * mu_Y + m * sigma_Y_sq * self.mu) / (m * sigma_Y_sq + sigma_prior_sq)
        sigma_new = np.sqrt((sigma_prior_sq * sigma_Y_sq) / (m * sigma_Y_sq + sigma_prior_sq))

        # Update estimates
        self.mu = mu_new
        self.sigma = sigma_new

        # Store history
        self.mu_history.append(self.mu.copy())
        self.sigma_history.append(self.sigma.copy())

    def get_distribution(self):
        """
        Returns the current mean and standard deviation estimates.
        """
        return self.mu, self.sigma

    def get_history(self):
        """
        Returns the history of mean and standard deviation estimates over time.
        """
        return np.array(self.mu_history), np.array(self.sigma_history)


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: GroundTruthDataset,
                 prior_computational_burden: BayesianNormalEstimator,
                 prior_fidelity: BayesianNormalEstimator,
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
        self._best_substitution_plan = None
        self._simulator = Simulator.Simulator()

        # Bayesian estimators for each substitution plan
        self.computational_burden_estimators = {}
        self.fidelity_estimators = {}

        for plan in self.repository.model_repository.keys():
            self.computational_burden_estimators[plan] = BayesianNormalEstimator(mu_prior=10, sigma_prior=5)
            self.fidelity_estimators[plan] = BayesianNormalEstimator(mu_prior=0.5, sigma_prior=0.2)

    def construct(self,
                  ):
        """
        Constructs the best GreyBox model using greedy optimization.
        :return: (best_plan, performance_history)
        """
        best_plan, history = self._greedy_optimizer()
        self._best_substitution_plan = best_plan  # Store the best plan
        return best_plan, history

    def substitution_plan_list(self):
        return [k for k, _ in self.repository.model_repository.items()]

    def get_best_model(self):
        """
        Returns the best model based on the performance criteria.
        """
        best_plan = None
        best_score = float("inf")

        for plan in self.repository.keys():
            mean_exec_time, _ = self.computational_burden_estimators[plan].get_distribution()
            mean_fidelity, _ = self.fidelity_estimators[plan].get_distribution()

            score = mean_exec_time + mean_fidelity  # Lower is better

            if score < best_score:
                best_score = score
                best_plan = plan

        return best_plan

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

        # Get substitution plans
        s_list = self.substitution_plan_list()

        # Select randomly the first substitution plan, ensuring it contains at least one `1`
        while current_s := np.random.choice(s_list):
            if 1 in current_s:
                break

        # Iterate through all batches
        for batch in tqdm.tqdm(batches):
            # Simulate the batch
            simulated_data_batch = self._opt_simulate_batch(current_s, batch)

            # Update estimators
            self._opt_update(current_s, simulated_data_batch)  # Every update stores history

            # Compute VoI for all substitution plans
            voi_values = self._opt_compute_voi(s_list)

            # Select the next plan based on VoI
            current_s = self._next_substitution_plan_heuristic(voi_values)

        # Select the final best model
        best_plan = self.get_best_model()
        history = self.get_history()
        return best_plan, history

    def _next_substitution_plan_heuristic(self, voi_values):
        """
        Selects the next substitution plan based on the computed VoI values.
        This should implement a heuristic such as greedy selection, epsilon-greedy, or Thompson Sampling.
        """
        # Example: Greedy selection (always picks the plan with highest VoI)
        return max(voi_values, key=voi_values.get)

    def _opt_simulate_batch(self,
                            substitution_plan,
                            batch,
                            ):
        """
        Simulates a batch of data using the given substitution plan.
        """
        # Get the model
        gbm = self.repository.get_model(substitution_plan)

        # Simulate
        params = Simulator.SimulationParameters(initial_time=batch["initial_time"],
                                                mission_time=batch["mission_time"],
                                                time_step=batch["time_step"],
                                                initial_state=batch["initial_state"],
                                                external_stimuli=batch["external_stimuli"],
                                                forced_states=batch["forced_states"],
                                                )
        self._simulator.params = params
        self._simulator.plant = gbm
        sim_data = self._simulator.simulate()

        return sim_data

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
