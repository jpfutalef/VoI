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
                           risk_metric,
                           ):
        """
        Update the computational burden and fidelity performance metrics for a substitution plan.

        :param plan: Substitution plan.
        :param sim_data_list: List of cictionaries of simulation data
        """
        # Measure the properties to update them
        z_l1 = computational_load(sim_data_list)
        z_l2 = lack_of_fit(gt_data_list,
                           sim_data_list,
                           risk_metric,
                           self.reference_plant,
                           self.get_model(plan),
                           )
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


def computational_load(sim_data_list):
    """
    Computes the computational load of a simulation data dictionary.

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
    for sim_data in sim_data_list:
        # Get necessary data
        t_sim = np.array(sim_data["time"])
        t_exec = np.array(sim_data["execution_time_array"])
        t_exec = t_exec - t_exec[0]
        m = slope_fit(t_sim, t_exec)
        results.append(m)

    return results


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

    # Compute the metric from each simulation
    metric_ref = [risk_metric(ref_sim_data, plant_ref) for ref_sim_data in ref_sim_data_list]
    metric = [risk_metric(sim_data, plant_gbm) for sim_data in sim_data_list]

    # Aggregate the metric (one per simulation)
    agg_metric_ref = [aggregate(x["time"], m) for x, m in zip(ref_sim_data_list, metric_ref)]
    agg_metric = [aggregate(x["time"], m) for x, m in zip(sim_data_list, metric)]

    # Turn into np array for KS computation
    agg_metric_ref = np.array(agg_metric_ref)
    agg_metric = np.array(agg_metric)

    # Compute KS statistic for the aggregated metric
    ks, info = ks_statistic(agg_metric_ref, agg_metric)

    return ks, info
