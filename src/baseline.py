import numpy as np

from base import BaseModel
from metric import compute_accuracy

from sknetwork.classification import PageRankClassifier, Propagation, DiffusionClassifier, NNClassifier


class Baseline(BaseModel):
    """Baseline model class."""
    def __init__(self, name: str):
        super(Baseline, self).__init__(name)
        if name == 'pagerank':
            self.alg = PageRankClassifier(solver='piteration')
        elif name == 'labelpropagation':
            self.alg = Propagation()
        elif name == 'diffusion':
            self.alg = DiffusionClassifier()
        elif name == 'knn':
            self.alg = NNClassifier()

    def get_seeds(self, labels_true: np.ndarray, train_idx: np.ndarray) -> dict:
        """Get training seeds in the form of a dictionary.
        
        Parameters
        ----------
        labels_true: np.ndarray
            True node labels.
        train_idx: np.ndarray
            Training indexes.
            
        Returns
        -------
            Dictionary of training seeds. """
        # Training data: corresponds to seed nodes
        training_seeds = {i: labels_true[i] for i in train_idx}
        return training_seeds
    
    def fit_predict(self, dataset, train_idx: np.ndarray) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Dataset object.
        train_idx: np.ndarray
            Training indexes.
            
        Returns
        -------
            Array of predicted node labels.
        """
        training_seeds = self.get_seeds(dataset.netset.labels_true, train_idx)

        labels_pred = self.alg.fit_predict(dataset.netset.adjacency, training_seeds)
        
        return labels_pred
    
    def accuracy(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool) -> float:
        """Accuracy score.
        
        Parameters
        ----------
        dataset
            Dataset object.
        labels_pred: np.ndarray
            Predicted labels.
        split: np.ndarray
            Split indexes.
        penalized: bool
            If true, labels not predicted (with value -1) are considered in the accuracy computation.
            
        Returns
        -------
            Accuracy score"""
        return compute_accuracy(dataset.netset.labels_true[split], labels_pred[split], penalized)
