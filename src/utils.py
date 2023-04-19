from base import BaseModel
from baseline import Baseline


def get_model(model: str) -> BaseModel:
    """Get model."""
    if model.lower() in ['pagerank']:
        return Baseline(model.lower())
    else:
        raise ValueError(f'Unknown model: {model}.')