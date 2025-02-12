"""
This module enables using a CPS Grey-Box Model based on a substitution plan.

Author: Juan-Pablo Futalef
"""
import copy
from typing import List
from itertools import product

from greyboxmodels.modelbuild import Plant


class GreyBoxRepository:
    def __init__(self,
                 reference_plant: Plant.HierarchicalPlant,
                 bbm_plants: List[Plant.Plant],
                 ):
        """
        A class to enable handy use of a Grey-Box Model.

        Parameters
        ----------
        reference_plant : Plant.HierarchicalPlant
            The reference White-Box Model Plant.
        bbm_plants : List[Plant.Plant]
            The Black-Box Model Plants.
            Length of the list must be equal to the number of plants in the reference_plant.

        Returns
        -------
        GreyBoxRepository

        """
        # Check if the number of plants in the reference model is equal to the number of plants in the BBM.
        assert len(reference_plant.plants) == len(
            bbm_plants), "Number of plants in the reference model must be equal to the number of plants in the BBM."

        # This plant inherits the properties of the reference plant.
        self.reference_plant = reference_plant
        self.bbm_plants = bbm_plants

        # Generate grey-box hierarchies
        self.model_repository, self.model_performance = self.generate_greybox_hierarchies()

    def generate_greybox_hierarchies(self):
        """
        Generate all possible grey-box hierarchies based on substitution plans.
        """
        repo = {}  # Dictionary to store models
        performance = {}  # Dictionary to store performances
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

            # Store the model using the substitution plan as the key
            repo[tuple(plan)] = gbm  # Convert list to tuple to use as key
            performance[tuple(plan)] = {}  # Initialize performance dictionary

        return repo, performance

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
        return self.model_repository[plan]
