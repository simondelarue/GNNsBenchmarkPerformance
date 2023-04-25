import os
import pickle
import sys

from base import BaseModel
from baseline import Baseline
from model import GNN


def get_model(model: str, dataset = None) -> BaseModel:
    """Get model."""
    if model.lower() in ['pagerank', 'labelpropagation', 'diffusion', 'knn']:
        return Baseline(model.lower())
    elif model.lower() in ['gcn', 'graphsage', 'gat', 'sgc']:
        return GNN(model.lower(), dataset)
    else:
        raise ValueError(f'Unknown model: {model}.')
    
def save_dict(path: str, filename: str, data: dict):
    """Save dictionary"""
    with open(f'{os.path.join(path, filename)}', 'wb') as f:
        pickle.dump(data, f)

def load_dict(path: str, filename: str) -> dict:
    """Load dictionary."""
    with open(f'{os.path.join(path, filename)}', 'rb') as f:
        data = pickle.load(f)
    return data

def check_exists(path: str, filename: str, force_run: bool = False):
    """Terminate program if file exists."""
    if not force_run and os.path.exists(os.path.join(path, filename)):
        sys.exit(f'File "{filename}" already exists.')
