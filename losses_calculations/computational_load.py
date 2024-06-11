from losses_calculations.setup_vars import *
from greyboxmodels.metrics import computational_load as metric

#%% Create a folder to store the values
target_dir = setup_dir(metric)

# %% Compute
table, info = metric.folders_comparison(folders, names, save_to=target_dir)
