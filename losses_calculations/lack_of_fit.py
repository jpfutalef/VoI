import os

import numpy as np
from pathlib import Path
import dill as pickle
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
import greyboxmodels.cpsmodels.Plant as Plant


def ks_statistic(ecdf_1, ecdf_2):
    """
    Compute the Kolmogorov-Smirnov statistic between two empirical cumulative distribution functions.

    Parameters
    ----------
    ecdf_1 : np.ndarray
        The first empirical cumulative distribution function.

    ecdf_2 : np.ndarray
        The second empirical cumulative distribution function.

    Returns
    -------
    float
        The Kolmogorov-Smirnov statistic.
    """
    # Remove NaNs
    ecdf_1 = ecdf_1.copy()[~np.isnan(ecdf_1)]
    ecdf_2 = ecdf_2.copy()[~np.isnan(ecdf_2)]
    return np.max(np.abs(ecdf_1 - ecdf_2))


def extract_state_dataframe(sim_path, exclude_from_col=None):
    """
    Open the simulation file and return the states.

    Parameters
    ----------
    sim_path : Path
        The path to the simulation file.

    exclude_from_col : int
        An int with the index of the columns to be excluded.

    Returns
    -------
    np.ndarray
        The states.
    """
    with open(sim_path, "rb") as f:
        d = pickle.load(f)

    df = pd.DataFrame(d["state"], index=d["time"])
    if exclude_from_col is not None:
        df = df.iloc[:, :exclude_from_col]

    return df


def states_from_folder(sim_folder, exclude_from_col=None):
    """
    Develops a list of dataframes with the states of the simulations.

    Parameters
    ----------
    sim_folder : Path
        The path to the folder with the simulations.

    exclude_from_col : int
        A list with the indexes of the columns to be excluded.

    Returns
    -------
    dict
        A list of dataframes with the states of the simulations.
    """
    locs = [x for x in sim_folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]

    state_dfs = {file.stem: extract_state_dataframe(file, exclude_from_col) for file in tqdm.tqdm(locs)}

    return state_dfs


# %% Test the approach
wbm_sim_folder = "D:/projects/CPS-SenarioGeneration/sim_data/monte_carlo/cpg/2024-03-20_18-55-20/"
wbm_sim_folder = Path(wbm_sim_folder)
wbm_states_loc = Path("sim_data/gbm-simulations/cpg") / "wbm_states.pkl"

# gbm_sim_folder = "sim_data/gbm-simulations/cpg/arch_1-0_1/2024-04-08_14-14-09"
# gbm_sim_folder = "sim_data/gbm-simulations/cpg/arch_2-1_0/2024-04-08_02-00-59"
gbm_sim_folder = "sim_data/gbm-simulations/cpg/arch_3-1_1/2024-04-08_02-00-59"
gbm_sim_folder = Path(gbm_sim_folder)
gbm_name = gbm_sim_folder.parent.name
gbm_states_loc = Path("sim_data/gbm-simulations/cpg") / f"{gbm_name}_states.pkl"

try:
    with open(wbm_states_loc, "rb") as f:
        wbm_states = pickle.load(f)

except FileNotFoundError:
    print("Creating the states at:", wbm_states_loc)
    wbm_states = states_from_folder(wbm_sim_folder, exclude_from_col=80)
    with open(wbm_states_loc, "wb") as f:
        pickle.dump(wbm_states, f)

try:
    with open(gbm_states_loc, "rb") as f:
        gbm_states = pickle.load(f)

except FileNotFoundError:
    print("Creating the states at:", gbm_states_loc)
    gbm_states = states_from_folder(gbm_sim_folder, exclude_from_col=80)
    with open(gbm_states_loc, "wb") as f:
        pickle.dump(gbm_states, f)

# %% Open the plant
with open(wbm_sim_folder / "plant.pkl", "rb") as f:
    wbm_plant = pickle.load(f)

with open(gbm_sim_folder / "plant.pkl", "rb") as f:
    gbm_plant = pickle.load(f)

# %% State names
pg_state_names = Plant.get_variables_names(wbm_plant.power_grid.state_idx)
cc_state_names = Plant.get_variables_names(wbm_plant.control_center.state_idx)

# %% Remove the keys that are not in both dictionaries
# TODO THIS SHOULD NOT BE NECESSARY
keys = set(wbm_states.keys()).intersection(gbm_states.keys())
wbm_states = {key: wbm_states[key] for key in keys}
gbm_states = {key: gbm_states[key] for key in keys}

# %% Compute the KS statistic for each state
ks_dict = {}
wbm_epdfs = {}
gbm_epdfs = {}
wbm_ecdfs = {}
gbm_ecdfs = {}
states = list(wbm_states.values())[0].columns

for state in states:
    # Iterate times
    ks_list = []
    wbm_epdf_list = []
    gbm_epdf_list = []
    wbm_ecdf_list = []
    gbm_ecdf_list = []
    for t in list(wbm_states.values())[0].index:
        # Accumulate the values of this state at this time for all simulations
        wbm_values = np.array([wbm_states[key].loc[t, state] for key in keys])
        gbm_values = np.array([gbm_states[key].loc[t, state] for key in keys])

        # Compute the ECDF for each set of values
        # First, ensure both have the same bins
        n_bins = 50
        bins = np.linspace(min(np.min(wbm_values), np.min(gbm_values)),
                           max(np.max(wbm_values), np.max(gbm_values)),
                           n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bins_diff = np.diff(bins)

        # Compute the empirical pdf
        wbm_epdf, _ = np.histogram(wbm_values, bins=bins, density=True)
        gbm_epdf, _ = np.histogram(gbm_values, bins=bins, density=True)

        # Empirical CDF
        wbm_ecdf = np.cumsum(wbm_epdf) * bins_diff
        gbm_ecdf = np.cumsum(gbm_epdf) * bins_diff

        # Add zero and one to the ECDF to ensure the KS statistic is computed correctly
        wbm_ecdf = np.concatenate(([0], wbm_ecdf, [1]))
        gbm_ecdf = np.concatenate(([0], gbm_ecdf, [1]))

        # In the bins, include new min and max values
        first_diff = bins_diff[0]
        last_diff = bins_diff[-1]
        bin_centers = np.concatenate(([bin_centers[0] - first_diff], bin_centers, [bin_centers[-1] + last_diff]))

        # Compute the KS statistic
        ks = ks_statistic(wbm_ecdf, gbm_ecdf)

        # Store the KS statistic and the EPDF and ECDF
        ks_list.append(ks)
        wbm_epdf_list.append((bin_centers, wbm_epdf))
        gbm_epdf_list.append((bin_centers, gbm_epdf))
        wbm_ecdf_list.append((bin_centers, wbm_ecdf))
        gbm_ecdf_list.append((bin_centers, gbm_ecdf))

    # Store the KS statistic for this state
    ks_dict[state] = ks_list
    wbm_epdfs[state] = wbm_epdf_list
    gbm_epdfs[state] = gbm_epdf_list
    wbm_ecdfs[state] = wbm_ecdf_list
    gbm_ecdfs[state] = gbm_ecdf_list

# %% Plot an EPDF
y = np.random.normal(0, 1, 5000)
y2 = np.random.normal(0.2, 1.2, 5000)

n_bins = 50
bins = np.linspace(min(np.min(y), np.min(y2)),
                   max(np.max(y), np.max(y2)),
                   n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bins_diff = np.diff(bins)

# Compute the empirical pdf
y_epf, _ = np.histogram(y, bins=bins, density=True)
y2_epdf, _ = np.histogram(y2, bins=bins, density=True)

# Empirical CDF
y_ecdf = np.cumsum(y_epf) * bins_diff
y2_ecdf = np.cumsum(y2_epdf) * bins_diff

# Add zero and one to the ECDF to ensure the KS statistic is computed correctly
y_ecdf = np.concatenate(([0], y_ecdf, [1]))
y2_ecdf = np.concatenate(([0], y2_ecdf, [1]))

# In the bins, include new min and max values
first_diff = bins_diff[0]
last_diff = bins_diff[-1]
bin_centers = np.concatenate(([bin_centers[0] - first_diff], bin_centers, [bin_centers[-1] + last_diff]))

# Compute the KS statistic
e_cdf_diff = np.abs(y_ecdf - y2_ecdf)
# Index of maximum
idx = np.argmax(e_cdf_diff)
ks = e_cdf_diff[idx]

# Plot both epdfs together
fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, dpi=600)

# Histograms for the EPDFs
axs[0].hist(y, bins=bins, density=True, alpha=0.5, color="b")
axs[0].hist(y2, bins=bins, density=True, alpha=0.5, color="r")
axs[0].set_title("Empirical PDF")

axs[1].plot(bin_centers, y_ecdf, "b", label="y")
axs[1].plot(bin_centers, y2_ecdf, "r", label="y2")
axs[1].set_title("Empirical CDF")

# KS statistic
axs[1].plot([bin_centers[idx], bin_centers[idx]], [y_ecdf[idx], y2_ecdf[idx]], "k")
axs[1].annotate(f"KS: {ks:.2f}", (bin_centers[idx], y_ecdf[idx]), xytext=(0.5, 0.4),
                textcoords="axes fraction", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))

# ggplot
plt.style.use("ggplot")
plt.show()

# %% Plot the KS statistics
n_states = len(states)
n_cols = 8
n_rows = n_states // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12.95 * 2, 5.85 * 2), constrained_layout=True,
                        sharex=True, sharey=True, dpi=600)
axs = axs.flatten()

t = list(wbm_states.values())[0].index / 3600

for i in range(n_cols * n_rows):
    ax = axs[i]
    if i >= n_states:
        ax.axis("off")
        continue
    state = states[i]
    ax.plot(t, ks_dict[state], "k")
    ax.set_title(pg_state_names[i])

    # If last row, set data_bottom_up label
    if i >= n_states - n_cols:
        ax.set_xlabel("Time [s]")

    # If first column, set y label
    if i % n_cols == 0:
        ax.set_ylabel("KS statistic")

    ax.set_xlim([0, t[-1]])

fig.show()

# %% Plot the ECDFs
n_states = len(states)
n_cols = 8
n_rows = n_states // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12.95 * 2, 5.85 * 2), constrained_layout=True,
                        sharey=True)
axs = axs.flatten()

t = list(wbm_states.values())[0].index / 3600

for i in range(n_cols * n_rows):
    ax = axs[i]
    if i >= n_states:
        ax.axis("off")
        continue
    state = states[i]
    ax.plot(wbm_ecdfs[state][0][0], wbm_ecdfs[state][0][1], "b", label="WBM")
    ax.plot(gbm_ecdfs[state][0][0], gbm_ecdfs[state][0][1], "r", label="GBM")
    ax.set_title(pg_state_names[i])

    # If first column, set y label
    if i % n_cols == 0:
        ax.set_ylabel("ECDF")

# Legend at the top
fig.legend(["WBM", "GBM"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.show()

# %% Compute the total KS statistic
total_ks = np.sum([np.sum(ks_arr) / np.size(ks_arr) for s, ks_arr in ks_dict.items()]) / len(ks_dict)

print(f"Total KS statistic for {gbm_name}: {total_ks}")
print(f"L2 for {gbm_name}: {total_ks / 0.0855}")



#%%%%%%%%%%%%%%%%%%%MAIN SCRIPT
from losses_calculations.setup_vars import *
from greyboxmodels.metrics import lack_of_fit as metric

#%% Create a folder to store the values
target_dir = setup_dir(metric)

# %% Compute
table, info = metric.folders_comparison(folders, names, save_to=target_dir)
