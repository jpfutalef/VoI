"""
This module enables using a CPS Grey-Box Model based on a substitution plan.

Author: Juan-Pablo Futalef
"""
import copy
from typing import List
from itertools import product
import numpy as np

from greyboxmodels.modelbuild import Plant


class GreyBoxRepository:
    def __init__(self,
                 reference_plant: Plant.HierarchicalPlant,
                 bbm_plants: List[Plant.Plant],
                 ):
        """
        A class to enable handy use of a Grey-Box Model.
        :param reference_plant: Reference plant.
        :param bbm_plants: List of Black-Box Models.
        """
        # Check if the number of plants in the reference model is equal to the number of plants in the BBM.
        assert len(reference_plant.plants) == len(
            bbm_plants), "Number of plants in the reference model must be equal to the number of plants in the BBM."

        # This plant inherits the properties of the reference plant.
        self.reference_plant = reference_plant
        self.bbm_plants = bbm_plants

        # Generate grey-box hierarchies
        self.model_repository = self.generate_greybox_models()
        self.model_performance = None
        self.reference_plan = list(self.model_repository.keys())[0]

    def generate_greybox_models(self):
        """
        Generate all possible grey-box hierarchies based on substitution plans.
        """
        repo = {}  # Dictionary to store models
        num_plants = len(self.reference_plant.plants)

        # Generate all possible binary substitution plans (tuples of 0s and 1s)
        all_plans = list(product([0, 1], repeat=num_plants))

        for plan in all_plans:
            # Create a new hierarchy from reference_plant
            gbm = copy.deepcopy(self.reference_plant)  # Deepcopy to prevent modifying the original

            # Modify the plants list based on the substitution plan
            for idx, val in enumerate(plan):
                if val == 1:  # Replace with BBM where plan[idx] is 1
                    gbm.plants[idx] = self.bbm_plants[idx]

            # Store the model and its metrics using the substitution plan as the key
            repo[tuple(plan)] = {"model": gbm,
                                 }

        return repo

    def get_model(self, plan):
        """
        Get the model based on the substitution plan.

        Parameters
        ----------
        plan : tuple
            Substitution plan to get the model.

        Returns
        -------
        Plant.HierarchicalPlant
            The Grey-Box Model based on the substitution plan.
        """
        return self.model_repository[plan]["model"]

    def __len__(self):
        return len(self.model_repository)

    def num_of_plants(self):
        return len(self.reference_plant.plants)


