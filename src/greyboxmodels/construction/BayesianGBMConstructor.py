"""
A class that implements different approaches for selecting the best GBM alternatives.
In general, simulations are cached in the disk and then, whenever needed, reused to reduce computational burden of
experiments.

Author: Juan-Pablo Futalef
"""

import os
import random
from pathlib import Path
from typing import Union, Dict, Tuple
import numpy as np
import tqdm

import greyboxmodels.construction.SimulationDataset as SimulationDataset
import greyboxmodels.construction.simulation_helper as sim
from greyboxmodels.construction import GreyBoxRepository
from greyboxmodels.construction import Loss, NIG


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository.Repository,
                 gt_data: SimulationDataset.SimulationDataset,
                 batch_size: int,
                 batch_accident_percentage: float,
                 n_exclusion_iterations: int,
                 wbm_statistics: Dict[str, Tuple[float, ...]],
                 NIG_priors_l1: Dict[tuple, tuple],
                 NIG_priors_l2: Dict[tuple, tuple],
                 risk_metric: callable,
                 work_dir: Union[Path, str],
                 ref_plan=None,
                 w1: float = 0.5,
                 w2: float = 0.5,
                 lambda1: callable = lambda x: x,
                 lambda2: callable = lambda x: x,
                 plan_names: Dict[tuple, str] = None
                 ):
        """
        Initialize the constructor.

        :param model_repository: Instance of GreyBoxRepository.
        :param gt_data: Ground truth simulation data.
        :param risk_metric: Callable risk metric function.
        :param work_dir: Working directory path to store simulation results.
        :param ref_plan: Reference plan; if not provided, the first available plan is used.
        :param ref_data: Optional reference simulation data.
        """
        self.repository = model_repository
        self.gt_data = gt_data
        self.batch_size = batch_size
        self.batch_accident_percentage = batch_accident_percentage
        self.n_exclusion_iterations = n_exclusion_iterations
        self.wbm_statistics = wbm_statistics
        self.NIG_priors_l1 = NIG_priors_l1
        self.NIG_priors_l2 = NIG_priors_l2
        self.risk_metric = risk_metric
        self.substitution_plans = list(self.repository.keys())
        self.ref_plan = ref_plan if ref_plan is not None else self.substitution_plans[0]
        self.plan_names = {s: f"F_{i}" for i, s in enumerate(self.substitution_plans)} if plan_names is None else plan_names
        self.best_model = None

        # Default weights and lambda values (these are now embedded in the loss functions)
        self.w1 = w1
        self.w2 = w2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Compute L0 using the WBM statistics
        wbm_l1_mean = self.wbm_statistics['l1_mean']
        wbm_l1_std = self.wbm_statistics['l1_std']
        wbm_l2_mean = self.wbm_statistics['l2_mean']
        wbm_l2_std = self.wbm_statistics['l2_std']
        self.L0 = (self.w1 * self.lambda1(wbm_l1_mean) + self.w2 * self.lambda2(wbm_l2_mean) +
                   np.sqrt(self.lambda1(wbm_l1_std) ** 2 + self.lambda2(wbm_l2_std) ** 2))

        # Store working directory
        self.work_dir = Path(work_dir)

    def L(self, Fi, Theta):
        from greyboxmodels.construction.Loss import aggregated_loss_risk as agg_loss
        # mu1, sigma1 = Theta[Fi][0]
        # mu2, sigma2 = Theta[Fi][1]
        # loss = self.w1 * self.lambda1(mu1) + self.w2 * self.lambda2(mu2)
        loss = agg_loss(Fi, Theta, self.w1, self.w2, self.lambda1, self.lambda2)
        return loss

    def construct(self, random_selection: bool = False) -> Tuple[tuple, Dict]:
        # Prepare the working directory
        os.makedirs(self.work_dir, exist_ok=True)

        # Setup up timer
        import time
        total_execution_time = 0.0
        t = time.time()

        # List of plans excluding the reference
        gbm_plans = [plan for plan in self.substitution_plans if plan != self.ref_plan]
        ref_plant = self.repository[self.ref_plan]

        # Partition GT data
        Bgt, ids = self.gt_data.GTBatches(self.risk_metric, ref_plant, self.batch_size, self.batch_accident_percentage)

        # Save the ids of each batch for reference
        ids_path = self.work_dir / "scenarios_accident_classification.json"
        with open(ids_path, "w") as f:
            import json
            json.dump(ids, f, indent=4)


        # Initialization
        excluded_plans = {}
        NIG_prior_l1 = {plan: [prior] for plan, prior in self.NIG_priors_l1.items()}
        NIG_prior_l2 = {plan: [prior] for plan, prior in self.NIG_priors_l2.items()}

        L0 = self.L0
        performance_info = {
            "iterations": [],
            "L0": L0,
        }

        # Main Loop
        for k, bk in enumerate(tqdm.tqdm(Bgt, desc="Iterating batches")):
            # Exclusion list
            plans_to_restore = []
            for plan, timer in excluded_plans.items():
                excluded_plans[plan] -= 1
                if excluded_plans[plan] <= 0:
                    plans_to_restore.append(plan)

            for plan in plans_to_restore:
                del excluded_plans[plan]

            # Prior Analysis
            current_benefits = {}
            Theta_current = {Fi: (NIG.NIGtoNormal(NIG_prior_l1[Fi][k]),
                                  NIG.NIGtoNormal(NIG_prior_l2[Fi][k])) for Fi in gbm_plans}
            for Fi in gbm_plans:
                current_benefits[Fi] = L0 - self.L(Fi, Theta_current)

            # Pick the best candidate
            included_plans = {Fi: current_benefits[Fi] for Fi in current_benefits.keys() if Fi not in excluded_plans}
            if random_selection:
                U_array = np.array([included_plans[Fi] for Fi in included_plans.keys()])
                offset = np.min(U_array) - 1e-6 if np.min(U_array) <= 0 else 0.0
                U_array = U_array - offset
                sum_U = np.sum(U_array)
                probabilities = U_array / sum_U
                selected_plan = random.choices(population=list(included_plans.keys()), weights=probabilities, k=1)[0]

            else:
                selected_plan = max(included_plans, key=included_plans.get)

            Fr = self.repository[selected_plan]

            # Record performance for this iteration
            performance_info["iterations"].append({
                "iteration": k,
                "U_star": max(current_benefits, key=current_benefits.get),
                "selected_plan": selected_plan,
                "current_benefits": current_benefits.copy(),
                "excluded_plans": list(excluded_plans.keys()),
                "current_priors_l1": {p: priors[-1] for p, priors in NIG_prior_l1.items()},
                "current_priors_l2": {p: priors[-1] for p, priors in NIG_prior_l2.items()}
            })

            total_execution_time += time.time() - t
            t = time.time()

            # Simulate batch
            scenarios = sim.simulate_plan(selected_plan, bk, self.repository, self.work_dir, parallel=True, pbar_offset=0)

            sim_exec_time = sum([x['total_execution_time'] for x in scenarios])
            total_execution_time += sim_exec_time

            # Collect evidence from the simulated batch
            z_l1, _ = Loss.computational_load(scenarios)
            z_l2, _ = Loss.lack_of_fit(bk, scenarios, self.risk_metric, ref_plant, Fr)

            # Update priors for the selected plan and keep the rest unchanged
            for Fi in gbm_plans:
                if Fi == selected_plan:
                    new_prior_l1 = NIG.NIGUpdate(NIG_prior_l1[Fi][-1], z_l1)
                    new_prior_l2 = NIG.NIGUpdate(NIG_prior_l2[Fi][-1], z_l2)

                else:
                    new_prior_l1 = NIG_prior_l1[Fi][-1]
                    new_prior_l2 = NIG_prior_l2[Fi][-1]

                NIG_prior_l1[Fi].append(new_prior_l1)
                NIG_prior_l2[Fi].append(new_prior_l2)


            # Add to Taboo List
            excluded_plans[selected_plan] = self.n_exclusion_iterations

        # Get the final GBM by using the last posterior
        Final_Theta = {}
        for Fi in gbm_plans:
            t1 = NIG.NIGtoNormal(NIG_prior_l1[Fi][-1])
            t2 = NIG.NIGtoNormal(NIG_prior_l2[Fi][-1])
            Final_Theta[Fi] = (t1, t2)

        final_utilities = {Fi: L0 - self.L(Fi, Final_Theta) for Fi in gbm_plans}
        final_utilities[self.ref_plan] = 0.0
        best_plan = max(final_utilities, key=final_utilities.get)

        total_execution_time += time.time() - t

        performance_info["final_best_plan"] = best_plan
        performance_info["final_utilities"] = final_utilities
        performance_info["prior_history_l1"] = NIG_prior_l1
        performance_info["prior_history_l2"] = NIG_prior_l2
        performance_info["total_execution_time"] = total_execution_time

        return best_plan, performance_info


