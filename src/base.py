from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit_predict():
        pass

    @abstractmethod
    def accuracy():
        pass