"""
This module enables using a CPS Grey-Box Model based on a substitution plan.

Author: Juan-Pablo Futalef
"""
import copy
from typing import List, Dict, Tuple
from itertools import product
from greyboxmodels.modelbuild import Plant

# Type alias for clarity
Repository = Dict[Tuple[int, ...], Plant.HierarchicalPlant]


def generate_greybox_repository(
        reference_plant: Plant.HierarchicalPlant,
        bbm_plants: List[Plant.Plant],
) -> Repository:
    """
    Generates a dictionary of Grey-Box Models based on all possible substitution plans.

    Args:
        reference_plant (Plant.HierarchicalPlant): The baseline White-Box/Reference plant.
        bbm_plants (List[Plant.Plant]): A list of Black-Box Models to substitute into the reference.

    Returns:
        Dict[Tuple[int, ...], Plant.HierarchicalPlant]: A dictionary where:
            - Keys are tuples representing the substitution plan (e.g., (0, 1, 0)).
            - Values are the resulting HierarchicalPlant objects (the GBMs).

    Raises:
        ValueError: If the number of sub-plants in the reference does not match the BBM list.
    """

    # 1. Validation
    if len(reference_plant.plants) != len(bbm_plants):
        raise ValueError(
            f"Mismatch: Reference model has {len(reference_plant.plants)} plants, "
            f"but {len(bbm_plants)} BBMs were provided."
        )

    repo = {}
    num_plants = len(reference_plant.plants)

    # 2. Generate all binary substitution plans (0=Reference, 1=BBM)
    # Result example: [(0,0), (0,1), (1,0), (1,1)]
    all_plans = list(product([0, 1], repeat=num_plants))

    # 3. Build models for each plan
    for plan in all_plans:
        # Deepcopy the reference to ensure we don't mutate the original input
        gbm = copy.deepcopy(reference_plant)

        # Iterate through the plan logic
        for idx, use_bbm in enumerate(plan):
            if use_bbm == 1:
                # Substitute the White-Box component with the Black-Box Model
                gbm.plants[idx] = bbm_plants[idx]

        # Map the plan tuple directly to the model object
        repo[tuple(plan)] = gbm

    return repo
