from base import BaseModel
import numpy as np

import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv
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
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=hidden_channels, aggr='max')
        self.conv2 = SAGEConv(hidden_channels, dataset.num_classes, aggr='max')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x
    

class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 8):
        super().__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels * 8, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

class SGC(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=2, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class GNN(BaseModel):
    """GNN model class."""
    def __init__(self, name: str, dataset, train_idx: np.ndarray):
        super(GNN, self).__init__(name)
        if name == "gcn":
            self.alg = GCN(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 200
        elif name == 'graphsage':
            self.alg = GraphSage(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 10
            self.train_loader = NeighborLoader(dataset.data, num_neighbors=[25, 10],
                                               batch_size=512, input_nodes=train_idx)
        elif name == 'gat':
            self.alg = GAT(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.05, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 100
        elif name == 'sgc':
            self.alg = SGC(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.2, weight_decay=0.005)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 100


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

        if val_idx is not None:
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
        if self.train_loader is not None:
            total_loss = 0
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                out = self.alg(batch.x, batch.edge_index)
                loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            loss = total_loss / len(self.train_loader)
        else:
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
        self.alg.train()
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
            return compute_accuracy(np.array(dataset.data.y[dataset.data.train_mask]), np.array(labels_pred[dataset.data.train_mask]), penalized)
        elif split_name == 'test':
            return self.test(dataset.data)
