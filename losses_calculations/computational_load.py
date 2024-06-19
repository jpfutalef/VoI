from losses_calculations.setup_vars import *
from greyboxmodels.metrics import computational_load as metric

#%% Create a folder to store the values
target_dir = setup_dir(metric)

# %% Compute
table, info = metric.folders_comparison(folders, names, save_to=target_dir)

#%% Some figures
import matplotlib.pyplot as plt
import dill as pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

info_path = target_dir / "computational_load_info.pkl"

with open(info_path, "rb") as f:
    info = pickle.load(f)

table = pd.read_csv(target_dir / "computational_load.csv", index_col=0, header=0)

# %% plot the time arrays
plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(4.58, 2.5))

# Plot realizations
min_t = 0
max_t = 0

# A condainer to store the m values
m_values = []

for i, ((folder, sims), m) in enumerate(zip(info.items(), table.itertuples())):
    # Get the color using i from discrete color palette
    color = plt.cm.tab10(i)

    for sim_name, sim_info in sims.items():
        if sim_name == "avg_load":
            continue

        # Get the data
        t = sim_info["sim_time"]
        exec_time = sim_info["exec_time"]

        # Update min and max
        min_t = min(min_t, t.min())
        max_t = max(max_t, t.max())

        # Plot
        alpha = 0.02
        ax.plot(t, exec_time, color=color, alpha=alpha)

    # Create line using m
    t = np.arange(min_t, max_t)
    y = m.L1 * t

    # Plot. label m in scientific notation
    ax.plot(t, y, label=f"M{i} (m={m.L1:.2e})",
            color=color, linewidth=2, linestyle="--")

    # Store m
    m_values.append(m.L1)

# Legend: sort the labels according to m.L1
handles, labels = ax.get_legend_handles_labels()
labels, handles, m_values = zip(*sorted(zip(labels, handles, m_values), key=lambda x: x[2]))
# Legend with two oclumns
plt.legend(handles, labels, loc="upper left", fontsize=6, ncol=2)

# lims
ax.set_xlim(0, max_t)
# ax.set_ylim(0, 10)
ax.set_ylim(0, 400)

# X ticks
x_ticks_pos = np.arange(0, max_t, 3600 * 8)
x_ticks_labels = [f"{int(x // 3600)}" for x in x_ticks_pos]
ax.set_xticks(x_ticks_pos)
ax.set_xticklabels(x_ticks_labels)
plt.xticks(rotation=-45)

# Labels
ax.set_xlabel("Scenario simulation time [h]")
ax.set_ylabel("Execution time [s]")

# Title
ax.set_title("Computational load (L1)")

# Save
plt.tight_layout()
plt.show()

#%% print figure size
fig.get_size_inches()
