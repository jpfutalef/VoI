import os
from losses_calculations.setup_vars import *
from greyboxmodels.voi.metrics import lack_of_fit as metric

# %% Create a folder to store the values
target_dir = setup_dir(metric)
reference_dir = Path(os.environ["VOI_ROOT"]) / "data/simulation_references/iptlc/MonteCarlo/2024-05-09_15-16-30/"


# %% The state filter
def state_filter(i_x):
    # Returns true if the state is to be dropped (filters out the state)
    if i_x > 79:
        # In the power grid, the first 80 states are the continuous states.
        # The rest are the discrete states and the variables from other devices.
        return True
    return False


# %% Compute
table, info = metric.folders_comparison(folders,
                                        reference_dir,
                                        target_dir,
                                        names,
                                        state_filter=state_filter,
                                        )

#%% Some figures
import matplotlib.pyplot as plt
import dill as pickle
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

info_loc = "data/voi_losses/lack_of_fit/2024-05-09_15-16-30/lack_of_fit_info.pkl"

with open(info_loc, "rb") as f:
    info = pickle.load(f)

#%% Get target values
model = 0
l = list(info.keys())
d = info[l[model]]["detailed_info"]
n_states = len(d)

#%% Create the figure
plt.close("all")

# Compute a nice number of rows and columns
n_cols = int(np.ceil(np.sqrt(n_states)))
n_rows = int(np.ceil(n_states / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8), sharex=True)
axs = axs.flatten()

for i_X, dinfo in d.items():
    t = [x[0] for x in dinfo]
    lof = [x[1] for x in dinfo]

    ax = axs[i_X]

    ax.plot(t, lof)
    ax.set_title(f"State {i_X}")
    ax.set_xlim([0, t[-1]])
    ax.set_ylim([0, 1])

    if i_X % n_cols == 0:
        ax.set_ylabel("LoF")

    if i_X >= n_states - n_cols:
        ax.set_xlabel("Time (s)")

# Remove the unused axes
for i_X in range(n_states, len(axs)):
    fig.delaxes(axs[i_X])

plt.tight_layout()
plt.show()
