from abc import ABC, abstractmethod
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold

import torch
from torch_geometric.data import Data


class BaseDataset:
    """Base class for Dataset."""

    def __init__(self, dataset: str, undirected: bool, random_state: int, k: int, stratified: bool):
        self.random_state = random_state
        self.data = self.get_data(dataset, undirected)
        self.kfolds = self.k_fold(self.data, k, random_state, stratified)
        self.netset = None

    def k_fold(self, data: Data, k: int, random_state: int, stratified: bool = True) -> tuple:
        """Split all data in Data into k folds. Each fold contains train/val/test splits, where val and test sizes equal 1/k.
        
        Parameters
        ----------
        data: Data
            torch.Data wrapper containing graph and feature information.
        k: int
            k in k-folds method.
        random_state: int
            Controls the reproducility.
        stratified: bool
            If True, use stratified kfold.
            
        Returns
        -------
            Tuple of train/val/test indices for each fold.
        """
        if stratified:
            skf = StratifiedKFold(k, shuffle=True, random_state=random_state)
        else:
            skf = KFold(k, shuffle=True, random_state=random_state)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(data.x)), data.y):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(k)]

        for i in range(k):
            train_mask = torch.ones(len(data.x), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        
        return (train_indices, test_indices, val_indices)
    

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