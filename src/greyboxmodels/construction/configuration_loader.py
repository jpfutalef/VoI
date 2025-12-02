import json
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional


def parse_tuple_key(key_str: str) -> Tuple[int, ...]:
    """
    Helper to convert string keys like "0,0,1" into tuples (0, 0, 1).
    """
    # Remove any accidental whitespace
    clean_str = key_str.replace(" ", "")
    return tuple(map(int, clean_str.split(',')))


@dataclass
class GBMConfig:
    """
    Data container for the entire GBM Selection process configuration.
    """
    model_library: Dict[Tuple[int, ...], str]
    reference_plan: Tuple[int, ...]
    wbm_statistics: Dict[str, Tuple[float, ...]]
    selection_parameters: Dict[str, Any]
    priors_l1: Dict[Tuple[int, ...], List[float]]
    priors_l2: Dict[Tuple[int, ...], List[float]]
    batch_size: int
    batch_accident_percentage: float
    n_exclusion_iterations: int
    num_predictive_data: int

    @classmethod
    def from_json(cls, json_path: str) -> 'GBMConfig':
        """
        Factory method to create a GBMConfig object from a JSON file.
        Automatically handles string-to-tuple conversion for keys.
        """
        try:
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {json_path}")

        # 1. Parse Model Library (String keys -> Tuple keys)
        raw_library = raw_data.get('model_library', {})
        model_library = {parse_tuple_key(k): v for k, v in raw_library.items()}

        # 2. Parse Reference Plan
        ref_str = raw_data.get('reference_plan', "0,0,0")
        reference_plan = parse_tuple_key(ref_str)

        # 3. Parse Priors (L1 and L2)
        # The JSON structure groups them under "priors" -> "l1", "l2"
        priors_data = raw_data.get('prior', {})

        raw_l1 = priors_data.get('computational_load', {})
        priors_l1 = {parse_tuple_key(k): v for k, v in raw_l1.items()}

        raw_l2 = priors_data.get('fidelity', {})
        priors_l2 = {parse_tuple_key(k): v for k, v in raw_l2.items()}

        # 4. Extract other sections directly
        wbm_statistics = raw_data.get('wbm_statistics', {})
        selection_parameters = raw_data.get('selection_parameters', {})

        # 5. extract the rest
        batch_size = raw_data.get('batch_size', 10)
        batch_accident_percentage = raw_data.get('batch_accident_percentage', 0.1)
        n_exclusion_iterations = raw_data.get('n_exclusion_iterations', 5)
        num_predictive_data = raw_data.get('num_predictive_data', 100)


        return cls(
            model_library=model_library,
            reference_plan=reference_plan,
            wbm_statistics=wbm_statistics,
            selection_parameters=selection_parameters,
            priors_l1=priors_l1,
            priors_l2=priors_l2,
            batch_size=batch_size,
            batch_accident_percentage=batch_accident_percentage,
            n_exclusion_iterations=n_exclusion_iterations,
            num_predictive_data=num_predictive_data
        )


def from_json(json_path: str) -> GBMConfig:
    """
    Convenience function to load GBMConfig from a JSON file.
    """
    return GBMConfig.from_json(json_path)