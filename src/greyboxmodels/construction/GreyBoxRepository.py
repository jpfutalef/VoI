"""
This module enables using a CPS Grey-Box Model based on a substitution plan.

Author: Juan-Pablo Futalef
"""
import copy
from typing import List
from itertools import product
import numpy as np

from greyboxmodels.modelbuild import Plant


class GreyBoxRepository:
    def __init__(self,
                 reference_plant: Plant.HierarchicalPlant,
                 bbm_plants: List[Plant.Plant],
                 ):
        """
        A class to enable handy use of a Grey-Box Model.
        :param reference_plant: Reference plant.
        :param bbm_plants: List of Black-Box Models.
        """
        # Check if the number of plants in the reference model is equal to the number of plants in the BBM.
        assert len(reference_plant.plants) == len(
            bbm_plants), "Number of plants in the reference model must be equal to the number of plants in the BBM."

        # This plant inherits the properties of the reference plant.
        self.reference_plant = reference_plant
        self.bbm_plants = bbm_plants

        # Generate grey-box hierarchies
        self.model_repository = self.generate_greybox_models()
        self.model_performance = None
        self.reference_plan = list(self.model_repository.keys())[0]

    def generate_greybox_models(self):
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

            # Initialize metrics for computational load and fidelity
            metrics = {
                "computational_load": Metric(),
                "fidelity": Metric()
            }

            # Store the model and its metrics using the substitution plan as the key
            repo[tuple(plan)] = {"model": gbm,
                                 "performance": metrics
                                 }

        return repo

    def prior_values(self,
                     simulation_data_list,
                     risk_metric,
                     num_scenarios=50,
                     batch_size=10,
                     ):
        """
        Returns the prior values of the Bayesian estimators.
        TODO find a way to include in init
        """
        # Get the indices of num_scenarios alternatives
        indices = np.random.choice(len(simulation_data_list), num_scenarios, replace=False)
        sim_data_list = [simulation_data_list[i] for i in indices]

        # Create a dictionary to store the prior performances
        perf = {plan: {} for plan in self.model_repository.keys()}

        """
        Computational burden
        """
        # Read the sim data and compute the average computational load
        z_l1, info_l1 = computational_load(sim_data_list)
        mu_l1_wbm = np.mean(z_l1)
        sigma_l1_wbm = np.std(z_l1, ddof=1)  # Sample standard deviation

        # For the BBMs, assume a computational burden 10% the mean of the reference model
        mu_l1_bbm = 0.1 * mu_l1_wbm
        sigma_l1_bbm = sigma_l1_wbm

        # Create priors for each plan: consider each subsystem is 1/3 of the total computational burden
        mu_l1_sub_wbm = mu_l1_wbm / 3
        mu_l1_sub_bbm = mu_l1_bbm / 3
        sigma_l1_sub_wbm = sigma_l1_wbm / 3
        sigma_l1_sub_bbm = sigma_l1_bbm / 3

        for plan in self.model_repository:
            mu_vals = [mu_l1_sub_bbm if x else mu_l1_sub_wbm for x in plan]
            sigma_vals = [sigma_l1_sub_bbm if x else sigma_l1_sub_wbm for x in plan]

            # Apply normal sum properties to obtain gbm mu and sigma
            mu_gbm = np.sum(mu_vals)
            sigma_gbm = np.sqrt(np.sum(np.square(sigma_vals)))

            # Store the estimators
            perf[plan]["computational_load"] = BayesianNormalEstimator(mu_gbm, sigma_gbm)

        """
        Fidelity
        """
        # Read the sim data and compute the average lack of fit
        mu_l2_wbm = 0.1
        sigma_l2_wbm = 0.1

        # For the BBMs, assume a lof 1.5 times that of the reference model
        mu_l2_bbm = 1.8 * mu_l2_wbm
        sigma_l2_bbm = 1.8 * sigma_l2_wbm

        # Create priors for each plan
        for plan in self.model_repository:
            mu_vals = [mu_l2_bbm if x else mu_l2_wbm for x in plan]
            sigma_vals = [sigma_l2_bbm if x else sigma_l2_wbm for x in plan]

            # Apply normal sum properties to obtain gbm mu and sigma
            mu_gbm = np.sum(mu_vals)
            sigma_gbm = np.sqrt(np.sum(np.square(sigma_vals)))

            # Store the estimators
            perf[plan]["lack_of_fit"] = BayesianNormalEstimator(mu_gbm, sigma_gbm)

        # Write to the internal attributes
        self.model_performance = perf

        return self.model_performance

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
        return self.model_repository[plan]["model"]

    def __len__(self):
        return len(self.model_repository)

    def num_of_plants(self):
        return len(self.reference_plant.plants)

    def update_performance(self,
                           plan,
                           sim_data_list,
                           gt_data_list,
                           risk_metric,
                           ):
        """
        Update the computational burden and fidelity performance metrics for a substitution plan.

        :param plan: Substitution plan.
        :param sim_data_list: List of cictionaries of simulation data
        """
        # Measure the properties to update them
        z_l1, info_l1 = computational_load(sim_data_list)
        z_l2, info_l1 = lack_of_fit(ref_sim_data_list=gt_data_list,
                                    sim_data_list=sim_data_list,
                                    risk_metric=risk_metric,
                                    plant_ref=self.reference_plant,
                                    plant_gbm=self.get_model(plan),
                                    )
        self.model_performance[plan]["computational_load"].update(z_l1, info_l1)
        self.model_performance[plan]["lack_of_fit"].update(z_l2, info_l1)

    def voi(self, loss_fun, **kwargs):
        """
        Computes Value of Information (VoI) for all substitution plans.
        """
        # Storage
        voi_dict = {}

        # Reference
        ref_loss = loss_fun(self.reference_plan, **kwargs)
        voi_dict[self.reference_plan] = {"loss": ref_loss, "voi": 0}

        # Compute the loss
        for plan in self.model_repository.keys():
            if plan == self.reference_plan:
                continue
            loss = loss_fun(plan, **kwargs)
            voi = ref_loss - loss
            voi_dict[plan] = {"loss": loss, "voi": voi}

        return voi_dict

    def plan_loss(self, plan, w1, w2):
        """
        Computes the model loss for a given plan.
        """

        # Get the performance data
        l1_estimator = self.model_performance[plan]["computational_load"]
        l2_estimator = self.model_performance[plan]["lack_of_fit"]

        # Get the mean and standard deviation of the performance metrics
        mu1, sigma1 = l1_estimator.get_mean_variance()
        mu2, sigma2 = l2_estimator.get_mean_variance()

        # Compute the loss
        loss = w1 * mu1 + w2 * mu2

        return loss

    def plan_loss_variance_penalized(self, plan, w1, w2, w3):
        """
        Computes the model loss for a given plan.
        """
        # Get the performance data
        l1_estimator = self.model_performance[plan]["computational_load"]
        l2_estimator = self.model_performance[plan]["lack_of_fit"]

        # Get the mean and standard deviation of the performance metrics
        mu1, sigma1 = l1_estimator.get_mean_variance()
        mu2, sigma2 = l2_estimator.get_mean_variance()

        # Scale the means
        mu1_scaled = mu1 * w1  # Normalize computational burden
        mu2_scaled = mu2 * w2  # Normalize fidelity

        # Scale the standard deviations using the same transformation
        sigma1_scaled = sigma1 * w1
        sigma2_scaled = sigma2 * w2

        # Compute the loss function
        loss = mu1_scaled + mu2_scaled + w3 * np.sqrt(sigma1_scaled ** 2 + sigma2_scaled ** 2)

        return loss
