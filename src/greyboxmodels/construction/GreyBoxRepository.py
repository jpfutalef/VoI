"""
This module enables using a CPS Grey-Box Model based on a substitution plan.

Author: Juan-Pablo Futalef
"""
import copy
from typing import List
from itertools import product
import numpy as np

from greyboxmodels.modelbuild import Plant


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

    # TODO generalize to any estimator or model performance metric
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

    def update(self, sim_data):
        """
        Performs a Bayesian update on the mean and standard deviation based on new observations.

        :param sim_data: Dictionary with simulation data.
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


class GreyBoxRepository:
    def __init__(self,
                 reference_plant: Plant.HierarchicalPlant,
                 bbm_plants: List[Plant.Plant],
                 ):
        """
        A class to enable handy use of a Grey-Box Model.

        Parameters
        ----------
        reference_plant : Plant.HierarchicalPlant
            The reference White-Box Model Plant.
        bbm_plants : List[Plant.Plant]
            The Black-Box Model Plants.
            Length of the list must be equal to the number of plants in the reference_plant.

        Returns
        -------
        GreyBoxRepository

        """
        # Check if the number of plants in the reference model is equal to the number of plants in the BBM.
        assert len(reference_plant.plants) == len(
            bbm_plants), "Number of plants in the reference model must be equal to the number of plants in the BBM."

        # This plant inherits the properties of the reference plant.
        self.reference_plant = reference_plant
        self.bbm_plants = bbm_plants

        # Generate grey-box hierarchies
        self.model_repository = self.generate_greybox_hierarchies()
        self.model_performance = {s: {"computational_burden": BayesianNormalEstimator(0, 1),
                                      "fidelity": BayesianNormalEstimator(0, 1)}
                                  for s in self.model_repository}

    def generate_greybox_hierarchies(self):
        """
        Generate all possible grey-box hierarchies based on substitution plans.
        """
        repo = {}  # Dictionary to store models
        num_plants = len(self.reference_plant.plants)

        # Generate all possible binary substitution plans (tuples of 0s and 1s)
        all_plans = list(product([0, 1], repeat=num_plants))

        for plan in all_plans:
            # Create a new hierarchy from reference_plant
            gbm = copy.deepcopy(self.reference_plant)  # Deepcopy to prevent modifying the original

            # Modify the plants list based on the substitution plan
            for idx, val in enumerate(plan):
                if val == 1:  # Replace with BBM where plan[idx] is 1
                    gbm.plants[idx] = self.bbm_plants[idx]

            # Store the model using the substitution plan as the key
            repo[tuple(plan)] = gbm  # Convert list to tuple to use as key

        return repo

    def get_model(self, plan):
        """
        Get the model based on the substitution plan.

        Parameters
        ----------
        plan : tuple
            Substitution plan to get the model.

        Returns
        -------
        Plant.HierarchicalPlant
            The Grey-Box Model based on the substitution plan.
        """
        return self.model_repository[plan]

    def __len__(self):
        return len(self.model_repository)

    def num_of_plants(self):
        return len(self.reference_plant.plants)

    def update_performance(self,
                           plan,
                           sim_data_list,
                           gt_data_list,
                           ):
        """
        Update the computational burden and fidelity performance metrics for a substitution plan.

        :param plan: Substitution plan.
        :param sim_data_list: List of cictionaries of simulation data
        """
        # Measure the properties to update them
        z_l1 = list(map(computational_load, sim_data_list))
        z_l2 = list(map(lack_of_fit, sim_data_list, gt_data_list))
        self.model_performance[plan]["computational_burden"].update(z_l1)
        self.model_performance[plan]["fidelity"].update(z_l2)

    def voi(self, plan=None):
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
UTILITY FUNCTIONS
"""


def computational_load(sim_data):
    """
    Computes the computational load of a simulation data dictionary.

    :param sim_data: Dictionary with simulation data.
    :return: The computational load of the simulation.
    """
    return 1  # Placeholder for now


def lack_of_fit(sim_data, ref_sim_data):
    """
    Computes the lack of fit between two simulation data dictionaries.

    :param ref_sim_data: Reference simulation data dictionary.
    :param sim_data: Simulation data dictionary.
    :return: The lack of fit between the two simulations.
    """
    return 1  # Placeholder for now
