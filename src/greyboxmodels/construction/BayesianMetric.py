"""
Set of tools to do Value of Information tasks.

Author: Juan-Pablo Futalef
"""
class Metric:
    def __init__(self, value=None):
        """
        Initializes the Metric class.

        :param value: A simple numeric value.
        """
        self.value = value

    def get(self):
        """
        Returns the metric value.

        :return: The metric value.
        """
        return self.value

    def update(self, value):
        """
        Updates the metric value.

        :param value: The new value to update.
        """
        self.value = value


class BayesianNormalEstimator(Metric):
    def __init__(self, mu_prior, sigma_prior):
        """
        Initializes the Bayesian normal estimator with prior mean and standard deviation.

        :param mu_prior: Prior mean (scalar or array for multiple variables).
        :param sigma_prior: Prior standard deviation (same shape as mu_prior).
        """
        super().__init__()
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

    def get(self):
        """
        Returns the current mean estimate.

        :return: The current mean estimate.
        """
        return self.mu

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




###
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