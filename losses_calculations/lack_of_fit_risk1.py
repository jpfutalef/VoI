"""
Computation of the lack of fit considering the state variables necessary for calculation of the following risk metric:
    METRIC: Lack of fit

Author: Juan-Pablo Futalef
"""

import os
from losses_calculations.setup_vars import *
from greyboxmodels.voi.metrics import lack_of_fit as metric

# %% Create a folder to store the values
target_dir = setup_dir(metric, subdirectory="risk_metric_1")
reference_dir = Path(os.environ["VOI_ROOT"]) / "data/simulation_references/iptlc/MonteCarlo/2024-05-09_15-16-30/"


#%% open plant
from greyboxmodels.cpsmodels.Plant import Plant
from greyboxmodels.cpsmodels.cyberphysical.IPTLC.IPTLC import IPandTLC

plant_dir = Path(os.environ["MODELS_ROOT"]) / "data/wbm-models/iptlc_ieee14-deterministic_tlcn7_wbm.pkl"
plant: IPandTLC = Plant.load(plant_dir)


# %% The state filter
def state_filter(i_x):
    import numpy as np
    # Only power demands in the state
    pd_idx = plant.power_grid.state_idx.Pd
    qd_idx = plant.power_grid.state_idx.Qd
    target_idx = np.concatenate([pd_idx, qd_idx])

    if i_x not in target_idx:
        return True
    return False


# %% Compute
table, info = metric.folders_comparison(folders,
                                        reference_dir,
                                        target_dir,
                                        names,
                                        state_filter=state_filter,
                                        )
