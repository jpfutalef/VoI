import pandas as pd
import matplotlib.pyplot as plt

#%% get table from clipboard
data = pd.read_clipboard(index_col=0)

#%% weights
weights = pd.read_clipboard(header=None, index_col=0)

#%% scaling factors
scaling_factors = pd.read_clipboard(header=None, index_col=0)

#%% Scale data> multiple element-wise the columns of the data table by the scaling factors and ignoring the index
data_scaled = data.mul(scaling_factors.values, axis=1)

#%% weighted costs
data_weighted = data_scaled.mul(weights.values, axis=1)

#%% Sum the weighted costs to get the weighted sum
data_weighted_sum = data_weighted.sum(axis=0)

#%% Voi: use the first column as the reference
voi = data_weighted_sum.iloc[0] - data_weighted_sum

#%% Cost-wise voi
individual_voi = -data_scaled.sub(data_scaled["A0"], axis=0)
# individual_voi.index = [f"{i}" for i in individual_voi.index]

#%% plot 3d VOI
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for voi_name, voi_values in individual_voi.iterrows():
    ax.scatter()

plt.show()

# %% VoI pareto
import pandas as pd
voi_vals = pd.read_clipboard(header=None, index_col=0)


