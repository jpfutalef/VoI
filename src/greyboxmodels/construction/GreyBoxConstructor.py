"""
Iterative method to guide GBM construction using VoI, coupled with some optimization algorithm.

Author: Juan-Pablo Futalef
"""
import copy
import random
from pathos.multiprocessing import ProcessPool
import dill as pickle
import tqdm
import multiprocessing as mp

from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository, computational_load, lack_of_fit
from greyboxmodels.construction.GroundTruth import GroundTruthDataset
from greyboxmodels.simulation import Simulator


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: GroundTruthDataset,
                 gt_batch_size: int = 5,
                 s0=None,
                 ):
        """
        This constructor enables the iterative selection of the best model from a repository of models
        and a dataset of ground truth data. The selection is guided by the Value of Information (VoI) metric.

        :param model_repository: A repository of grey-box models
        :param gt_data: Ground truth data
        :param params: Constructor parameters
        """
        self.repository = model_repository
        self.gt_data = copy.deepcopy(gt_data)
        self.gt_batch_size = gt_batch_size

        # Elements to be used during the optimization process
        self._s0 = s0
        self._best_substitution_plan = None
        self._simulator = Simulator.Simulator()
        self._const_current_plan = None
        self.substitution_plans = list(self.repository.model_repository.keys())

    def construct(self,
                  method="greedy_voi",
                  **kwargs,
                  ):
        """
        Constructs the best GreyBox model using greedy optimization.
        :return: (best_plan, performance_history)
        """
        methods = {"greedy_voi": self._greedy_optimizer,
                   "exhaustive_numerical": self._exhaustive_optimizer,
                   }
        try:
            optimizer = methods[method]
        except KeyError:
            raise ValueError("Method not found. Available methods are: {}".format(methods.keys()))

        return optimizer(**kwargs)

    def get_best_model(self):
        """
        Returns the best model based on the performance criteria.
        """
        return

    def get_history(self):
        """
        Returns historical performance metrics.
        """
        history = {"computational_burden": {}, "fidelity": {}}

        for plan in self.repository.keys():
            history["computational_burden"][plan] = self.computational_burden_estimators[plan].get_history()
            history["fidelity"][plan] = self.fidelity_estimators[plan].get_history()

        return history

    def _exhaustive_optimizer(self, w1=1., w2=1., n_scenarios=5, parallel=True):
        """
        Computes the performances of all substitution plans and selects the best one.
        If `parallel=True`, executes in parallel using multiple processes.
        """
        perf = {"method": "exhaustive_numerical"}

        # Extract a shuffled subset of scenarios
        scenarios = self.gt_data.extract(n_scenarios)

        if parallel:
            manager = mp.Manager()
            models_data = manager.dict(self.repository.model_repository)
            with ProcessPool() as pool:
                results = pool.map(GreyBoxModelConstructor._process_plan,
                                   self.substitution_plans,
                                   [scenarios] * len(self.substitution_plans),
                                   [models_data] * len(self.substitution_plans),
                                   )

            for plan, result in results:
                perf[plan] = result
        else:
            for plan in self.substitution_plans[5:]:
                plan, result = self._process_plan(plan, scenarios, self.repository)
                perf[plan] = result

        # Obtain the best model
        sums = {plan: w1 * p["computational_burden"] + w2 * p["fidelity"] for plan, p in perf.items()}
        best_model = min(sums, key=sums.get)

        return best_model, perf

    def _greedy_optimizer(self):
        """
        Performs a greedy optimization process for selecting the best GreyBox substitution plan.
        """
        # Get the batches
        batches = self.gt_data.get_batches(self.gt_batch_size)

        # Select a first substitution plan, the heuristic knows how to do it
        current_s = self._next_substitution_plan_heuristic() if self._s0 is None else self._s0

        # Iterate through all batches
        for batch in tqdm.tqdm(batches):
            # Simulate the batch
            simulated_data_batch = self._opt_simulate_batch(current_s, batch, self.repository)

            # Update estimators
            self.repository.update_performance(current_s, simulated_data_batch)  # Every update stores history

            # Select the next plan to simulate
            current_s = self._next_substitution_plan_heuristic()

        # Select the final best model
        best_plan = self.get_best_model()
        history = self.get_history()
        return best_plan, history

    def _next_substitution_plan_heuristic(self):
        """
        The heuristic uses VoI to select the next substitution plant in the iteration
        """
        # Check start
        if self._best_substitution_plan is None:
            valid_S = [S for S in self.substitution_plans if any(S)]
            S = random.choice(valid_S)
            return S

        # From the repository, use the meand and std to compute the relative error
        # Then, use the relative error to compute the VoI
        # Finally, select the plan with the highest VoI
        pass

    def _opt_compute_voi(self):
        """
        Computes Value of Information (VoI) for all substitution plans.
        """
        subs_list = self.substitution_plan_list()
        Sref = subs_list[0]  # Reference plan (WBM)
        voi_values = {}
        for plan in subs_list[1:]:
            # Get those values for the reference plan (WBM)
            ref_mean_comp_burd, ref_std_comp_burd = self.computational_burden_estimators[Sref].get_distribution()
            ref_mean_fidelity, ref_std_fidelity = self.fidelity_estimators[Sref].get_distribution()

            # Get those values for the current plan
            mean_comp_burd, std_comp_burd = self.computational_burden_estimators[plan].get_distribution()
            mean_fidelity, std_fidelity = self.fidelity_estimators[plan].get_distribution()

            # Aggregate each
            Lref = ref_mean_comp_burd + ref_mean_fidelity
            L = mean_comp_burd + mean_fidelity

            # Compute VoI
            VoI_plan = Lref - L

            # Store
            voi_values[plan] = VoI_plan

        return voi_values

    @staticmethod
    def _process_plan(plan, scenarios, repository):
        """Simulates all scenarios for a given plan (must be picklable for multiprocessing)."""
        gbm_outputs = GreyBoxModelConstructor._opt_simulate_batch(plan, scenarios, repository)

        # Compute performance metrics
        l1 = computational_load(gbm_outputs)
        l2 = lack_of_fit(scenarios, gbm_outputs)

        return plan, {"computational_burden": l1, "fidelity": l2, "simulated_data": gbm_outputs}

    @staticmethod
    def _opt_simulate_batch(substitution_plan, scenarios, repository):
        """
        A static version of `_opt_simulate_batch()` that can be called in parallel.
        It retrieves the model and runs the simulation.
        """
        gbm = repository.get_model(substitution_plan)  # Use the repository instance
        gbm.stochastic = False  # Disable stochasticity always

        results = []
        for scenario in scenarios:
            params = Simulator.SimulationParameters(
                initial_time=scenario["initial_time"],
                mission_time=scenario["mission_time"],
                time_step=scenario["time_step"],
                initial_state=scenario["initial_state"],
                external_stimuli=scenario["external_stimuli"],
                forced_states=scenario["forced_states"],
            )
            simulator = Simulator.Simulator()
            simulator.params = params
            simulator.plant = gbm
            sim_data = simulator.simulate(pbar_post=f"S={substitution_plan}")
            results.append(sim_data)

        return results


"""
Utility functions
"""


def load(path):
    with open(path, "rb") as f:
        gc = pickle.load(f)

    return gc


def save(gc, path):
    with open(path, "wb") as f:
        pickle.dump(gc, f)
