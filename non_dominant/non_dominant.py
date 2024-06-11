"""
Script plots the three dimensional solutions to analyse the existence of non-dominant solutions.

Authhor: Juan-Pablo Futalef
"""
# %% Pareto library
# %pip install paretoset
# %pip install OApackage

# %% Change matplotlib backend
import matplotlib as mpl

# mpl.use('Qt5Agg')
mpl.use('TkAgg')
#mpl.use('Agg')

# %% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset, paretorank
import oapackage

# %% Matlab engine
import matlab.engine
# eng = matlab.engine.start_matlab()
eng = matlab.engine.start_matlab('-desktop')

# eng_names = matlab.engine.find_matlab()
# eng = matlab.engine.connect_matlab(eng_names[0])

# %% The table of results
data = pd.read_csv('data/numerical-voi/iptlc/voi-results.csv', index_col=0, header=[0, 1]).T

# %% filter by "losses"
data = data.loc[data.index.get_level_values(0) == 'losses', :]
data.index = data.index.droplevel(0)


# %% Pareto set using paretoset library
# pareto_membership = paretoset(data, sense=["min", "min", "min"], use_numba=False)
# pareto_rank = paretorank(data, sense=["min", "min", "min"], use_numba=False)

# %% Pareto set using oapackage library
# pareto=oapackage.ParetoDoubleLong()
#
# for ii in range(0, data.values.shape[1]):
#     w = oapackage.doubleVector( (data.values[0,ii], data.values[1,ii], data.values[2,ii]) )
#     pareto.addvalue(w, ii)
#
# # pareto.show(verbose=1)
#
# pareto_elements = list(pareto.allindices())
# pareto_membership = np.zeros(data.shape[0], dtype=bool)
# pareto_membership[pareto_elements] = True

# %% Compute the dominant solutions
# pareto_membership = np.ones(data.shape[0], dtype=bool)
#
# for i in range(data.shape[0]):
#     for j in range(data.shape[0]):
#         if i != j:
#             if (data.iloc[i, 0] > data.iloc[j, 0] and data.iloc[i, 1] > data.iloc[j, 1] and data.iloc[i, 2] > data.iloc[j, 2]):
#                 pareto_membership[i] = False
#                 break

# %% A function to compute the dominant solutions
# def check_dominance(obs1, obs2):
#     # obs1 = [l1_1, l2_1, l3_1, ...]
#     # obs2 = [l1_2, l2_2, l3_2, ...]
#     better_or_equal = obs1 <= obs2
#     better = obs1 < obs2
#     if np.all(better_or_equal) and np.any(better):
#         return True
#     return False
#
#
# pareto_membership = [True] * data.shape[0]
# for i in range(data.shape[0]):
#     # Iterate all the solutions
#     is_dominated = False
#     for j in range(data.shape[0]):
#         if i == j:
#             # Skip the same solution
#             continue
#
#         # Check if i is dominated by j
#         dominates = check_dominance(data.iloc[j, :], data.iloc[i, :])
#
#         # If i doesn't dominate j, break the loop
#         if not dominates:
#             break
#
#     # Write the result
#     pareto_membership[i] = dominates

#%% Very slow for many datapoints.  Fastest for many costs, most readable
# based on : https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def pareto_front(costs, op='min'):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

pareto_membership = pareto_front(data.values)

# %% Select the columns
x = data.loc[:, "l1"].values
y = data.loc[:, "l2"].values
z = data.loc[:, "l3"].values

# %%  The three dimensional plot
plt.close('all') # Close all previous figures

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, (is_pareto, arch_name) in enumerate(zip(pareto_membership, data.index)):
    if is_pareto:
        ax.scatter(x[i], y[i], z[i], c='b', marker='o')
    else:
        ax.scatter(x[i], y[i], z[i], c='r', marker='x')

    # Annotation
    ax.text(x[i], y[i], z[i], arch_name, color='black')

# Create a surface to join the points in the Pareto set
x_pareto = x[pareto_membership]
y_pareto = y[pareto_membership]
z_pareto = z[pareto_membership]

# ax.plot_trisurf(x_pareto, y_pareto, z_pareto, color='grey', alpha=0.4, antialiased=True)

# Labels
ax.set_xlabel('L1 (computational load)')
ax.set_ylabel('L2 (lack of fit)')
ax.set_zlabel('L3 (substitution cost)')

# Make axes have arrow heads

# Legend
ax.scatter([], [], c='b', marker='o', label='Pareto')
ax.scatter([], [], c='r', marker='x', label='Non-Pareto')
ax.legend()

# Tight layout
fig.tight_layout()

# Show
plt.show()

#%% Pass to matlab
m_costs = matlab.double(data.values.astype(float))
m_l1 = matlab.double(x.astype(float))
m_l2 = matlab.double(y.astype(float))
m_l3 = matlab.double(z.astype(float))
m_pareto_membership = matlab.logical(pareto_membership)

eng.workspace["costs"] = m_costs
eng.workspace["l1"] = m_l1
eng.workspace["l2"] = m_l2
eng.workspace["l3"] = m_l3
eng.workspace["pareto_membership"] = m_pareto_membership

#%% L1 vs L2
eng.scatter(m_l1, m_l2)

#
# width, height = 6, 3
# try:
#     fig = plt.gcf()
#     # set size
#     fig.set_size_inches(width, height)
#
#     ax = plt.gca()
# except:
#     fig, ax = plt.subplots(figsize=(width, height))
#
# for i, (is_pareto, arch_name) in enumerate(zip(pareto_membership, data.index)):
#     if is_pareto:
#         ax.scatter(x[i], y[i], c='b', marker='o')
#     else:
#         ax.scatter(x[i], y[i], c='r', marker='x')
#
#     # Annotation
#     ax.text(x[i], y[i], arch_name, color='black')
#
# # Labels
# ax.set_xlabel('L1 (computational load)')
# ax.set_ylabel('L2 (lack of fit)')
#
# # Title
# ax.set_title('L1 vs L2')
#
# fig.tight_layout()
# fig.show()

#%% Plot costs for each architecture
plt.close('all') # Close all previous figures
fig, axs = plt.subplots(3, 1, figsize=(12, 4))

for i, arch_name in enumerate(data.index):
    # bar plots
    axs[0].bar(arch_name, data.loc[arch_name, "l1"], color='b' if pareto_membership[i] else 'r')
    axs[1].bar(arch_name, data.loc[arch_name, "l2"], color='b' if pareto_membership[i] else 'r')
    axs[2].bar(arch_name, data.loc[arch_name, "l3"], color='b' if pareto_membership[i] else 'r')

# Title
axs[0].set_title('L1 (computational load)')
axs[1].set_title('L2 (lack of fit)')
axs[2].set_title('L3 (substitution cost)')

# Labels
axs[0].set_ylabel('L1')
axs[1].set_ylabel('L2')
axs[2].set_ylabel('L3')

# Tight layout
fig.tight_layout()
fig.show()


# %% VoI instead of costs
voi_table = pd.read_clipboard(index_col=0, header=0).T

#%% pareto for the voi table, but using max
def pareto_front(costs, op='min'):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

pareto_membership = pareto_front(voi_table.values, op='max')

# %% Select the columns
x = voi_table.loc[:, "voi1"].values
y = voi_table.loc[:, "voi2"].values
z = voi_table.loc[:, "voi3"].values

# %%  The three dimensional plot
plt.close('all') # Close all previous figures

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, (is_pareto, arch_name) in enumerate(zip(pareto_membership, voi_table.index)):
    if is_pareto:
        ax.scatter(x[i], y[i], z[i], c='b', marker='o')
    else:
        ax.scatter(x[i], y[i], z[i], c='r', marker='x')

    # Annotation
    ax.text(x[i], y[i], z[i], arch_name, color='black')

# create lims
x_min_lim, x_max_lim = np.min(x), np.max(x)
y_min_lim, y_max_lim = np.min(y), np.max(y)
z_min_lim, z_max_lim = np.min(z), np.max(z)

min_val = min(x_min_lim, y_min_lim, z_min_lim)
max_val = max(x_max_lim, y_max_lim, z_max_lim)

# Create surfaces to show the X, Y, Z axes
ax.plot([x_min_lim, x_max_lim], [0, 0], [0, 0], '--', c='grey')
ax.plot([0, 0], [y_min_lim, y_max_lim], [0, 0], '--', c='grey')
ax.plot([0, 0], [0, 0], [z_min_lim, z_max_lim], '--', c='grey')

# limits
ax.set_xlim(x_min_lim, x_max_lim)
ax.set_ylim(y_min_lim, y_max_lim)
ax.set_zlim(z_min_lim, z_max_lim)

# Labels
ax.set_xlabel('VoI1 (computational load)')
ax.set_ylabel('VoI2 (lack of fit)')
ax.set_zlabel('VoI3 (substitution cost)')

# Legend
ax.scatter([], [], c='b', marker='o', label='Pareto')
ax.scatter([], [], c='r', marker='x', label='Non-Pareto')
ax.legend()

# Tight layout
fig.tight_layout()

# Show
plt.show()

# %% Plot all the VoIs individually
# latex
plt.rc('text', usetex=True)
plt.close('all') # Close all previous figures
fig, axs = plt.subplots(3, 1, figsize=[5.64,4.9])

for i, arch_name in enumerate(voi_table.index):
    # bar plots
    axs[0].bar(arch_name, voi_table.loc[arch_name, "voi1"], color='b' if pareto_membership[i] else 'r')
    axs[1].bar(arch_name, voi_table.loc[arch_name, "voi2"], color='b' if pareto_membership[i] else 'r')
    axs[2].bar(arch_name, voi_table.loc[arch_name, "voi3"], color='b' if pareto_membership[i] else 'r')

# Title
axs[0].set_title('VoI1 (computational load)')
axs[1].set_title('VoI2 (lack of fit)')
axs[2].set_title('VoI3 (substitution cost)')

# Labels
axs[0].set_ylabel('VoI1')
axs[1].set_ylabel('VoI2')
axs[2].set_ylabel('VoI3')

# Horizontal lines at zero
for ax in axs:
    ax.axhline(0, color='black', linewidth=0.5)

# Align the y labels
fig.align_ylabels()

# Tight layout
fig.tight_layout()

# Show
plt.show()

#%% Get figure size
fig = plt.gcf()
fig.get_size_inches()

print(fig.get_size_inches())
