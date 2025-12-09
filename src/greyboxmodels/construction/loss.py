import numpy as np


def aggregate_risk(t, r):
    # Integration
    return np.trapz(r, t)

def computational_load(scenario_list):
    """
    Compute the computational load for a list of scenarios.
    :param scenario_list: List of dictionaries with simulation data.
    :return: Tuple (mean, variance, detailed_info)
    """
    def slope_fit(t_sim, t_exec):
        t_sim_col = t_sim[:, np.newaxis]
        m, _, _, _ = np.linalg.lstsq(t_sim_col, t_exec, rcond=None)
        return m[0]

    evidence = []
    info = {"t_sim": [], "t_exec": []}
    for sim_data in scenario_list:
        t_sim = np.array(sim_data["time"])
        t_exec = np.array(sim_data["execution_time_array"]) - sim_data["execution_time_array"][0]
        m = slope_fit(t_sim, t_exec)
        evidence.append(m)
        info["t_sim"].append(t_sim)
        info["t_exec"].append(t_exec)

    mean_result = np.mean(evidence, dtype=np.float64)
    variance_result = np.var(evidence, dtype=np.float64)
    info_dict = {"results": evidence, "mean": mean_result, "variance": variance_result, "details": info}
    return evidence, info_dict


def lack_of_fit(ref_scenario_list, scenario_list, risk_metric, ref_plant, gbm_plant):
    """
    Compute the lack of fit between two simulation datasets using a risk metric.

    :param ref_scenario_list: List of reference simulation data dictionaries.
    :param scenario_list: List of simulation data dictionaries from the grey-box model.
    :param risk_metric: Callable risk metric used for comparing simulations.
    :param ref_plant: Reference plant model.
    :param gbm_plant: Grey-box model.
    :return: Tuple ((mean, variance), detailed_info)
    """

    def ks_statistic(data_ref, data, n_bins=20):
        if np.array_equal(data_ref, data):
            return 0, {"equal_data": True,
                       "ks_value": 0}
        min_x = min(data.min(), data_ref.min())
        max_x = max(data.max(), data_ref.max())
        bins = np.linspace(min_x, max_x, n_bins + 1)
        epdf_ref, _ = np.histogram(data_ref, bins=bins, density=True)
        epdf, _ = np.histogram(data, bins=bins, density=True)
        ecdf_ref = empirical_cdf(bins, epdf_ref)
        ecdf = empirical_cdf(bins, epdf)
        abs_diff = np.abs(ecdf - ecdf_ref)
        ks_value = np.max(abs_diff)
        max_diff_idx = np.argmax(abs_diff)
        info = {"ks_idx": max_diff_idx, "ks_value": ks_value, "ks_bin_loc": bins[max_diff_idx],
                "bins": bins, "epdf": epdf, "epdf_ref": epdf_ref, "ecdf": ecdf, "ecdf_ref": ecdf_ref,
                "abs_diff": abs_diff}
        return ks_value, info

    def empirical_cdf(bin_edges, epdf):
        cdf_values = np.cumsum(epdf * np.diff(bin_edges))
        cdf_values /= cdf_values[-1]
        return np.concatenate(([0], cdf_values))

    metric_ref = [risk_metric(ref, ref_plant) for ref in ref_scenario_list]
    metric = [risk_metric(sim, gbm_plant) for sim in scenario_list]
    agg_metric_ref = np.array([aggregate_risk(x["time"], m) for x, m in zip(ref_scenario_list, metric_ref)])
    agg_metric = np.array([aggregate_risk(x["time"], m) for x, m in zip(scenario_list, metric)])
    ks, info_dict = ks_statistic(agg_metric_ref, agg_metric)
    evidence = ks
    return evidence, info_dict

def aggregated_loss_risk(Fi, Theta, w1, w2, lambda1, lambda2):
    mu1, sigma1 = Theta[Fi][0]
    mu2, sigma2 = Theta[Fi][1]
    loss = w1 * lambda1(mu1) + w2 * lambda2(mu2) + np.sqrt(lambda1(sigma1) ** 2 + lambda2(sigma2) ** 2)
    return loss

def aggregated_loss_expected(Fi, Theta, w1, w2, lambda1, lambda2):
    mu1, sigma1 = Theta[Fi][0]
    mu2, sigma2 = Theta[Fi][1]
    loss = w1 * lambda1(mu1) + w2 * lambda2(mu2)
    return loss