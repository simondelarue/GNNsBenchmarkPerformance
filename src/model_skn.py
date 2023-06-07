from base import BaseModel
import numpy as np

from sknetwork.gnn import GNNClassifier

from metric import compute_accuracy


class GNNSkn(BaseModel):
    """GNN model class for scikit-network."""
    def __init__(self, name: str, dataset, train_idx: np.ndarray):
        super(GNNSkn, self).__init__(name)
        if name == 'gcn_skn':
            self.alg = GNNClassifier(dims=[16, dataset.data.num_classes],
                                     layer_types='Conv',
                                     normalizations='Both',
                                     activations='ReLu',
                                     optimizer='Adam', learning_rate=0.01,
                                     verbose=False)
            self.n_epochs = 200
            print(self.alg)

    def update_masks(self, data, train_idx, val_idx, test_idx):
        """Update train/val/test masks in netset Dataset object.
        
        Parameters
        ----------
        data: Torch Dataset
            Torch Dataset object.
        train_idx: np.ndarray
            Training indexes.
        val_idx: np.ndarray
            Validation indexes.
        test_idx: np.ndarray
            Test indexes.
        """
        labels = data.labels_true.copy()
        train_mask = np.zeros(len(labels)).astype(bool)
        train_mask[train_idx] = True
        
        # No val set
        test_mask = np.zeros(len(labels)).astype(bool)
        test_mask[test_idx] = True

        labels[test_mask] = -1

        # Create train and test mask in Data object
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask

        return labels
        
    def fit_predict(self, dataset, train_idx: np.ndarray, val_idx: np.ndarray, test_idx : np.ndarray, **kwargs) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Custom Dataset object.
            
        Returns
        -------
            Array of predicted node labels.
        """
        # Build train/val/test masks
        labels = self.update_masks(dataset.netset, train_idx, val_idx, test_idx)

        # Train model
        pred = self.alg.fit_predict(dataset.netset.adjacency, dataset.netset.biadjacency, labels, n_epochs=self.n_epochs + 1)

        return pred

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
