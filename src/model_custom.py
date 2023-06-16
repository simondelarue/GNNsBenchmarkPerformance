from typing import Optional
import numpy as np
from scipy import sparse

from base import BaseModel
from metric import compute_accuracy

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch.nn.init import xavier_uniform_
from torch_geometric.utils import to_dense_adj

# Imports from doc
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None, trainable_self_loops=None):

    fill_value = 2. if improved else 1.

    if trainable_self_loops is not None:
        fill_value = torch.sigmoid(trainable_self_loops)

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)
        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # Edge weight
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
        
   
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='add')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    #print(edge_index)
    #print(edge_weight)
    #print('Edge weights before adding self-loops', edge_weight, len(edge_weight))
    # Add self loops
    #print(f'Initial edge index: {len(edge_index[0])} - {len(edge_index[1])}')
    if add_self_loops:
        edge_index, _ = add_self_loops_fn(
            edge_index, edge_weight, 1, num_nodes)
        #edge_weight = torch.concat((torch.ones(edge_index.size(1) - num_nodes), fill_value)) 
        edge_weight = torch.concat((edge_weight, fill_value))
    #print(f'Used edge index: {len(edge_index[0])} - {len(edge_index[1])}')
    #print(f'Used weights : {edge_weight} - {len(edge_weight)} ')

    return edge_index, edge_weight


class GraphConvolution2(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # TODO: initialize with what?
        
        # Xavier initialization
        #xavier_stddev = 1 / math.sqrt(self.num_nodes)
        #self.weight0 = Parameter(torch.rand(self.num_nodes) * xavier_stddev)
        
        # Xavier uniform initialization
        weights = torch.empty((self.num_nodes,))
        xavier_uniform_(weights.unsqueeze(0))
        self.weight0 = Parameter(weights.squeeze(0))

        # Around 1 initialization, with He std
        #self.weight0 = Parameter(normal_(torch.empty((self.num_nodes,)), mean=1, std=(1 / math.sqrt(self.num_nodes))))
        
        
        print(f'Initial weight0: {self.weight0} - {len(self.weight0)}')

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype,
                        self.weight0)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class GraphConvolution(MessagePassing):
    """Code from https://github.com/asarigun/la-gcn-pytorch/blob/main/layers.py#L16"""

    def __init__(self, num_nodes, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters are assigned as nn.Module attributes in order to get automatically added to the list of its Parameters.
        self.weight0 = Parameter(torch.rand(num_nodes))
        self.bias0 = Parameter(torch.Tensor(num_nodes))
        self.lin = Linear(in_features, out_features, bias=False,
                          weight_initializer='glorot')
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        #for i in self.parameters():
        #    print(i)

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        zeros(self.bias0)

    def forward(self, X, edge_index):
        adjacency = to_dense_adj(edge_index)[0]

        # Left normalization of adjacency
        degrees = torch.matmul(adjacency, torch.ones(adjacency.shape[1]))
        degrees_inv = 1 / degrees
        degrees_inv[degrees_inv == float('inf')] = 0
        norm = torch.diag(degrees_inv)

        # Add trainable self-loops
        #adjacency_trainable = adjacency + torch.diag(torch.round(self.weight0))
        
        #adjacency_trainable = adjacency + torch.diag((self.weight0))
        
        adjacency_trainable = adjacency + torch.diag(2 * torch.sigmoid(self.weight0))

        #adjacency_trainable = adjacency + torch.diag(torch.ones(adjacency.shape[0]))

        # D^-1(A + W)
        adjacency_norm = torch.mm(norm, adjacency_trainable)
        #adjacency_norm_2 = torch.mm(adjacency_norm, norm)
    
        # Message passing
        support = torch.mm(adjacency_norm, X)
        #support = torch.mm(adjacency_norm_2, X)
        output = self.lin(support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNCustom(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int=16):
        super().__init__()

        # Graph convolutions
        #self.conv1 = GraphConvolution(dataset.x.shape[0], dataset.num_features,dataset.num_classes)
        #self.conv1 = GraphConvolution(dataset.x.shape[0], dataset.num_features, hidden_channels)
        #self.conv2 = GraphConvolution(dataset.x.shape[0], hidden_channels, dataset.num_classes)

        # From doc
        num_nodes = dataset.x.shape[0]
        self.conv1 = GraphConvolution2(num_nodes, dataset.num_features, hidden_channels)
        self.conv2 = GraphConvolution2(num_nodes, hidden_channels, dataset.num_classes)

        # Only 1 layer
        #self.conv1 = GraphConvolution2(num_nodes, dataset.num_features, dataset.num_classes)        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GNNCustom(BaseModel):
    """Custom GNN."""
    def __init__(self, name: str, dataset, train_idx: np.ndarray):
        super(GNNCustom, self).__init__(name)

        if name == 'gnn_custom':
            self.alg = GCNCustom(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 200

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
        self.alg.train()
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
            print('before criterion: ', self.alg.conv1.weight0)
            loss = self.criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            self.optimizer.step()
            print('after criterion: ', self.alg.conv1.weight0)

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
    
    def transform_data(self, dataset, **kwargs):
        """Apply transformation on data according to parameters.
        
        Parameters
        ----------
        dataset
            Dataset object.
        
        Returns
        -------
        Transformed data.
        """
        # Use concatenation of adjacency and features matrix 
        if kwargs.get('use_concat') == 'true':
            X = torch.tensor(sparse.hstack((dataset.netset.adjacency, dataset.netset.biadjacency)).todense(), dtype=torch.float32)
            dataset.data.x = X
            dataset.num_features = dataset.data.x.shape[1]

        return dataset
    
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
        dataset.data = self.update_masks(dataset.data, train_idx, val_idx, test_idx)

        # Train model
        losses = []
        trained_weights0 = []
        trained_weights1 = []
        for epoch in range(1, self.n_epochs + 1):
            
            #print('epoch ', epoch)
            #print('In epoch: ', self.alg.conv1.weight0, self.alg.conv1.weight0.sum())
            #print('In epoch: ', self.alg.conv2.weight0, self.alg.conv2.weight0.sum())
            #if epoch == 3:
            #    raise Exception('END')
                
            loss = self.train(dataset.data)
            losses.append(loss.item())
            trained_weights0.append(self.alg.conv1.weight0.detach().numpy().sum())
            trained_weights1.append(self.alg.conv1.lin.weight.detach().numpy().sum())
            
            #train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]
            #train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())
        #test_acc = self.test(self.alg, dataset)
        out = self.alg(dataset.data.x, dataset.data.edge_index)
        pred = out.argmax(dim=1)

        #fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        #plt.plot(range(self.n_epochs), trained_weights0, label='diag weights')
        #plt.plot(range(self.n_epochs), trained_weights1, label='model weights')
        #plt.legend()
        #plt.show()
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