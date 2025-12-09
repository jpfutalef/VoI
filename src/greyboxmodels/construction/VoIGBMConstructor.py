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
                 num_predictive_data: int,
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
        self.num_predictive_data = num_predictive_data
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
        wbm_l2_mean = self.wbm_statistics['l2_mean']
        self.L0 = self.w1 * self.lambda1(wbm_l1_mean) + self.w2 * self.lambda2(wbm_l2_mean)

        # Store working directory
        self.work_dir = Path(work_dir)

    def L(self, Fi, Theta):
        from greyboxmodels.construction.Loss import aggregated_loss_expected as agg_loss
        #mu1, sigma1 = Theta[Fi][0]
        #mu2, sigma2 = Theta[Fi][1]
        #loss = self.w1 * self.lambda1(mu1) + self.w2 * self.lambda2(mu2)
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

            # Current global best (U*)
            #U_star_current = max(0, max(current_benefits.values()))
            U_star_current = max(current_benefits.values())

            # Compute VoI for candidates not in the Exclusion List
            VoI = {}
            for Fi in gbm_plans:
                expected_pre_posterior_U_star = self.estimate_pre_posterior(Fi,
                                                                       Theta_current,
                                                                       U_star_current,
                                                                       NIG_prior_l1[Fi][k],
                                                                       NIG_prior_l2[Fi][k],
                                                                       L0,
                                                                       n_samples=self.num_predictive_data,
                                                                       n_samples_l1=len(bk),
                                                                       n_samples_l2=1,)
                VoI[Fi] = expected_pre_posterior_U_star - U_star_current

            # Pick the best candidate
            included_VoI = {Fi: VoI[Fi] for Fi in VoI.keys() if Fi not in excluded_plans}
            if random_selection:
                voi_array = np.array([included_VoI[Fi] for Fi in included_VoI.keys()])
                sum_voi = np.sum(voi_array)
                probabilities = voi_array / sum_voi
                selected_plan = random.choices(population=list(included_VoI.keys()), weights=probabilities, k=1)[0]

            else:
                selected_plan = max(included_VoI, key=included_VoI.get)

            Fr = self.repository[selected_plan]

            # Record performance for this iteration
            performance_info["iterations"].append({
                "iteration": k,
                "U_star": U_star_current,
                "VoI_values": VoI.copy(),
                "VoI_included": included_VoI.copy(),
                "selected_plan": selected_plan,
                "current_benefits": current_benefits.copy(),
                "excluded_plans": list(excluded_plans.keys()),
                "current_priors_l1": {p: priors[-1] for p, priors in NIG_prior_l1.items()},
                "current_priors_l2": {p: priors[-1] for p, priors in NIG_prior_l2.items()}
            })
            total_execution_time += time.time() - t
            t = time.time()

            # Simulate batch
            scenarios = sim.simulate_plan(selected_plan, bk, self.repository, self.work_dir, parallel=True,
                                          pbar_offset=0)

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

    def estimate_pre_posterior(self, Fi, currentTheta, U_star_current, prior_l1, prior_l2, L0, n_samples, n_samples_l1,
                               n_samples_l2):
        """
        Computes the expected pre-posterior utility of collecting data of GBM Fi.
        """
        sum_U_prime = 0.0
        theta_shadow = currentTheta.copy()
        for m in range(n_samples):
            # Emulate data collection
            Z_l1 = NIG.sample_predictive(prior_l1, n_samples=n_samples_l1)
            Z_l2 = NIG.sample_predictive(prior_l2, n_samples=n_samples_l2)

            # Emulate posterior update with the emulated data
            beta_prime_l1 = NIG.NIGUpdate(prior_l1, Z_l1)
            beta_prime_l2 = NIG.NIGUpdate(prior_l2, Z_l2)

            # Get Normal parameters from the updated NIG parameters
            theta_prime_l1 = NIG.NIGtoNormal(beta_prime_l1)
            theta_prime_l2 = NIG.NIGtoNormal(beta_prime_l2)

            # Calculate the utility of Fi if the experiment were run
            theta_shadow[Fi] = (theta_prime_l1, theta_prime_l2)

            # Collect all utilities
            U_tilde = [L0 - self.L(Fi, theta_shadow) for Fi in currentTheta.keys()]

            # Select the maximum expected utility
            U_prime_winner = max(*U_tilde)

            sum_U_prime += U_prime_winner

        # Average and Return
        expected_pre_posterior_U = sum_U_prime / n_samples

        return expected_pre_posterior_U

