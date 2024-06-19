from pathlib import Path

# %% Specify paths
wbm = Path("/mnt/d/projects/CPS-SenarioGeneration/data/iptlc/MonteCarlo/2024-05-09_15-16-30").resolve()
gbm1 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_1-0_0_1/2024-05-09_15-16-30").resolve()
gbm2 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_2-1_0_0/2024-05-09_15-16-30").resolve()
gbm3 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_3-1_0_1/2024-05-09_15-16-30").resolve()
gbm4 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_4-0_1_0/2024-05-09_15-16-30").resolve()
gbm5 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_5-0_1_1/2024-05-09_15-16-30").resolve()
gbm6 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_6-1_1_0/2024-05-09_15-16-30").resolve()
gbm7 = Path("/mnt/d/projects/IPTLC_BBMs/data/gbm-simulations/iptlc/arch_7-1_1_1/2024-05-09_15-16-30").resolve()

# %% Add to list
folders = [wbm, gbm1, gbm2, gbm3, gbm4, gbm5, gbm6, gbm7]
names = ["WBM" if i == 0 else f"GBM{i}" for i, _ in enumerate(folders)]


# %% Function to setup the path
def setup_dir(module):
    import os
    root = Path(os.environ["VOI_ROOT"])
    mod_name = module.__name__.split(".")[-1]
    target_dir = root / "data" / "voi_losses" / mod_name / wbm.name

    if target_dir.exists():
        print(f"WARNING: The target directory already exists. It will be overwritten.\n   {target_dir}")

    os.makedirs(target_dir, exist_ok=True)
    return target_dir


