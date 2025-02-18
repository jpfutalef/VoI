"""
This module enables using a CPS Grey-Box Model based on a substitution plan.

Author: Juan-Pablo Futalef
"""
import copy
from typing import List
from itertools import product
import numpy as np
from scipy.integrate import simpson

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

    def __init__(self,
                 mu_prior,
                 sigma_prior,
                 ):
        """
        Initializes the Bayesian normal estimator with prior mean and standard deviation.

        :param mu_prior: Prior mean (scalar or array for multiple variables).
        :param sigma_prior: Prior standard deviation (same shape as mu_prior).
        """
        self.mu = mu_prior
        self.sigma = sigma_prior

        # Track history for visualization
        self.mu_history = [self.mu]
        self.sigma_history = [self.sigma]

        # Track info history
        self.info_history = []

    def update(self, observations, info=None):
        """
        Performs a Bayesian update on the mean and standard deviation based on multiple observations of a single variable.

        :param observations: 1D array of new observations (multiple samples of a single variable).
        :param info: Optional metadata to store with update history.
        """
        observations = np.array(observations, dtype=np.float64)  # Ensure NumPy array
        m = len(observations)  # Number of new samples

        if m == 0:
            return  # No update if no new data

        # Compute sample mean and variance from new observations
        mu_Y = np.mean(observations)
        sigma_Y = np.std(observations, ddof=1)

        # Convert to variance for computation
        sigma_prior_sq = self.sigma ** 2
        sigma_Y_sq = max(sigma_Y ** 2, 1e-8)  # Avoid zero variance issues


        # Bayesian update formulas
        sigma_new_sq = (1 / sigma_prior_sq + m / sigma_Y_sq) ** -1
        mu_new = sigma_new_sq * (self.mu / sigma_prior_sq + m * mu_Y / sigma_Y_sq)

        # Convert variance back to standard deviation
        sigma_new = np.sqrt(sigma_new_sq)

        # Update estimates
        self.mu = mu_new
        self.sigma = sigma_new

        # Store history
        self.mu_history.append(self.mu)
        self.sigma_history.append(self.sigma)
        self.info_history.append(info)

    def get_mean_variance(self):
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
        :param reference_plant: Reference plant.
        :param bbm_plants: List of Black-Box Models.
        :param risk_metric: Risk metric to use for lack of fit.
        """
        # Check if the number of plants in the reference model is equal to the number of plants in the BBM.
        assert len(reference_plant.plants) == len(
            bbm_plants), "Number of plants in the reference model must be equal to the number of plants in the BBM."

        # This plant inherits the properties of the reference plant.
        self.reference_plant = reference_plant
        self.bbm_plants = bbm_plants

        # Risk metric to use for lack of fit
        # self.risk_metric = risk_metric    # TODO generalize
        # self.risk_aggregation = risk_aggregation  # TODO generalize

        # Generate grey-box hierarchies
        self.model_repository = self.generate_greybox_hierarchies()
        self.model_performance = None
        self.reference_plan = list(self.model_repository.keys())[0]

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


"""
UTILITY FUNCTIONS
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

    return results, info


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
