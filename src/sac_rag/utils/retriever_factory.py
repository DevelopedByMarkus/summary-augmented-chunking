from typing import Union
from sac_rag.data_models import RetrievalMethod
from sac_rag.methods.baseline import BaselineRetrievalMethod, RetrievalStrategy as BaselineConfig
from sac_rag.methods.hybrid import HybridRetrievalMethod, HybridStrategy

# A type hint for any of our strategy configuration models
AnyRetrievalConfig = Union[BaselineConfig, HybridStrategy]


def create_retriever(strategy_config: AnyRetrievalConfig) -> RetrievalMethod:
    """
    Factory function to create a retriever instance from a loaded strategy config.

    This function inspects the type of the configuration object and returns the
    corresponding retriever class, initialized with that configuration.

    Args:
        strategy_config: A validated Pydantic model for a retrieval strategy.

    Returns:
        An initialized instance of a class that inherits from RetrievalMethod.
    """
    if isinstance(strategy_config, BaselineConfig):
        print("Factory: Creating BaselineRetrievalMethod...")
        return BaselineRetrievalMethod(retrieval_strategy=strategy_config)

    elif isinstance(strategy_config, HybridStrategy):
        print("Factory: Creating HybridRetrievalMethod...")
        return HybridRetrievalMethod(strategy=strategy_config)

    else:
        # This case should ideally not be hit if the config loader works correctly,
        # but it's good defensive programming.
        raise TypeError(f"Unhandled strategy configuration type: {type(strategy_config)}")
