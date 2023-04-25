from base import BaseModel
import numpy as np

import torch
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F

from metric import compute_accuracy


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 16):
        super().__init__()
        #torch.manual_seed(0)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

class GraphSage(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 256):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=hidden_channels, aggr='max', project=False)
        self.conv2 = SAGEConv(hidden_channels, dataset.num_classes, aggr='max', project=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x
    

class GNN(BaseModel):
    """GNN model class."""
    def __init__(self, name: str, dataset):
        super(GNN, self).__init__(name)
        if name == "gcn":
            self.alg = GCN(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 100
        elif name == 'graphsage':
            self.alg = GraphSage(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 10


    def update_masks(self, data, train_idx, val_idx, test_idx):
        """Update train/val/test mask in Torch Dataset object.
        
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
            
        Returns
        -------
            Updated Torch Dataset object.
        """
        train_mask = torch.zeros(len(data.x), dtype=bool)
        train_mask[train_idx] = True
        data.train_mask = train_mask
        test_mask = torch.zeros(len(data.x), dtype=bool)
        test_mask[test_idx] = True
        
        test_mask[val_idx] = True # ---> no val set
        data.test_mask = test_mask

        return data
    
    def train(self, dataset) -> torch.Tensor:
        """Training function.
        
        Parameters
        ----------
        dataset: Custom Dataset object.
        
        Returns
        -------
            Loss. 
        """
        self.alg.train()
        self.optimizer.zero_grad()
        out = self.alg(dataset.x, dataset.edge_index)
        loss = self.criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        self.optimizer.step()
        
        return loss

    def test(self, data):
        """Test function.
        
        Parameters
        ----------
        data: Torch Data object.
        
        Returns
        -------
            Loss.
        """
        self.alg.eval()
        out = self.alg(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        
        return test_acc
    
    def fit_predict(self, dataset, train_idx: np.ndarray, val_idx: np.ndarray, test_idx : np.ndarray) -> np.ndarray:
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
        dataset.data = self.update_masks(dataset.data, train_idx, val_idx, test_idx)

        # Train model
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train(dataset.data)
            out = self.alg(dataset.data.x, dataset.data.edge_index)
            pred = out.argmax(dim=1)
            #train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]
            #train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())
        #test_acc = self.test(self.alg, dataset)

        return pred
    
    def accuracy(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool, split_name: str) -> float:
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
        split_name: str
            Either 'train' or 'test'.
            
        Returns
        -------
            Accuracy score"""
        if split_name == 'train':
            return compute_accuracy(dataset.data.y[dataset.data.train_mask], labels_pred[dataset.data.train_mask], penalized)
        elif split_name == 'test':
            return self.test(dataset.data)
