from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Base class for models."""

    def __init__(self, name: str):
        self.name = name
        self.train_loader = None

    @abstractmethod
    def fit_predict(self, dataset, train_idx: np.ndarray = None):
        pass

    @abstractmethod
    def accuracy(dataset, labels_pred, split, penalized, *args):
        pass