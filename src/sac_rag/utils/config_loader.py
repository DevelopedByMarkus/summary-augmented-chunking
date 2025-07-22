import json
from typing import Union
from pydantic import ValidationError

from sac_rag.methods.baseline import RetrievalStrategy as BaselineStrategy
from sac_rag.methods.hybrid import HybridStrategy

# A type hint for any of our possible strategy models
AnyRetrievalStrategy = Union[BaselineStrategy, HybridStrategy]


def load_strategy_from_file(filepath: str) -> AnyRetrievalStrategy:
    """
    Loads a retrieval strategy configuration from a JSON file and validates it
    using the appropriate Pydantic model.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not read or parse the config file at {filepath}")
        raise e

    strategy_type = data.get("strategy_type")

    # Add 'method_name' to data if it's HybridStrategy, which requires it
    if strategy_type == "hybrid":
        model = HybridStrategy
        if "method_name" not in data:
            data["method_name"] = "hybrid" # Default value
    elif strategy_type == "baseline":
        model = BaselineStrategy
    else:
        raise ValueError(
            f"Invalid 'strategy_type' in {filepath}: '{strategy_type}'. "
            "Must be 'baseline' or 'hybrid'."
        )

    try:
        # Pydantic does the magic of parsing and validating the nested dictionary
        return model.model_validate(data)
    except ValidationError as e:
        print(f"Error: Configuration file {filepath} failed validation.")
        raise e
