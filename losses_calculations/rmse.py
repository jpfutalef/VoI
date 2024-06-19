import os
from losses_calculations.setup_vars import *
from greyboxmodels.metrics import rmse as metric

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
