import numpy as np

from base import BaseModel
from metric import compute_accuracy

from sklearn.linear_model import LogisticRegression

from sknetwork.classification import PageRankClassifier, Propagation, DiffusionClassifier, NNClassifier
from sknetwork.embedding import Spectral


class Baseline(BaseModel):
    """Baseline model class."""
    def __init__(self, name: str, **kwargs):
        super(Baseline, self).__init__(name)
        if name == 'pagerank':
            self.alg = PageRankClassifier(solver='piteration')
        elif name == 'labelpropagation':
            self.alg = Propagation()
        elif name == 'diffusion':
            self.alg = DiffusionClassifier()
        elif name == 'knn':
            if kwargs.get('embedding_method') == 'true':
                self.alg = NNClassifier(n_neighbors=5, embedding_method=Spectral(30))
            else:
                self.alg = NNClassifier()
        elif name == 'logistic_regression':
            self.alg = LogisticRegression()

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
        training_seeds = {i.item(): labels_true[i] for i in train_idx}
        return training_seeds
    
    def fit_predict(self, dataset, train_idx: np.ndarray, val_idx: np.ndarray = None, test_idx : np.ndarray = None, **kwargs) -> np.ndarray:
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
        # Logistic regression from Sklearn does not have a fit_predict method
        if self.name == 'logistic_regression':
            if kwargs.get('use_features') == 'true':
                labels_pred = self.alg.fit(dataset.netset.biadjacency[train_idx, :], dataset.netset.labels_true[train_idx]).predict(dataset.netset.biadjacency)
            else:
                labels_pred = self.alg.fit(dataset.netset.adjacency[train_idx, :], dataset.netset.labels_true[train_idx]).predict(dataset.netset.adjacency)
        else:
            training_seeds = self.get_seeds(dataset.netset.labels_true, train_idx) 

            if kwargs.get('use_features') == 'true':
                labels_pred = self.alg.fit_predict(dataset.netset.biadjacency, training_seeds)
            else:
                labels_pred = self.alg.fit_predict(dataset.netset.adjacency, training_seeds)
        
        return labels_pred
    
    def accuracy(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool, *args) -> float:
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
