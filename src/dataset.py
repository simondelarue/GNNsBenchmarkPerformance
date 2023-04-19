import numpy as np
import os
import pickle
from scipy import sparse

from sklearn.model_selection import StratifiedKFold

from sknetwork.data import load_netset, Bunch

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class PlanetoidDataset:
    """Citation networks: Cora, PubMed, Citeseer."""

    def __init__(self, dataset: str, undirected: bool, random_state: int, k: int):
        self.undirected = undirected
        self.random_state = random_state
        self.data = self.get_data(dataset, self.undirected)        
        self.kfolds = self.k_fold(self.data, k, random_state)
        self.netset = None

    def get_netset(self, dataset: str, pathname: str, use_cache: bool = True):
        """Get data in Netset format (scipy.sparse CSR matrices). Save data in Bunch format if use_cache is set to False.
        
        Parameters
        ----------
        dataset: str
            Dataset name
        pathname: str
            Path to data.
        use_cache: bool (default=True)
            If True, use cached data (if existing).

        Returns
        -------
            Bunch object.
        """
        if os.path.exists(os.path.join(pathname, dataset)) and use_cache:
            with open(os.path.join(pathname, dataset), 'br') as f:
                graph = pickle.load(f)
            print(f'Loaded dataset from {os.path.join(pathname, dataset)}')
        else:
            print(f'Building netset data...')
            # Convert dataset to NetSet format (scipy CSR matrices)
            graph = self.to_netset()

            # Save Netset dataset
            with open(os.path.join(pathname, dataset), 'bw') as f:
                pickle.dump(graph, f)
            print(f'Netset data saved in {os.path.join(pathname, dataset)}')
        
        self.netset = graph
        
        return self.netset
    
    def to_netset(self):
        """Convert data into Netset format and return Bunch object."""
        # nodes and edges
        rows = np.asarray(self.data.edge_index[0])
        cols = np.asarray(self.data.edge_index[1])
        data = np.ones(len(rows))
        n = len(set(rows).intersection(set(cols)))
        adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        # Features
        biadjacency = sparse.csr_matrix(np.array(self.data.x), dtype=bool)

        # Node information
        labels = np.array(self.data.y)

        graph = Bunch()
        graph.adjacency = adjacency
        graph.biadjacency = biadjacency
        graph.labels = labels

        return graph

    def get_data(self, dataset: str, undirected: bool):
        """Get dataset information.
        
        Parameters
        ----------
        dataset: str
            Dataset name
        undirected: bool
            If True dataset is forced to undirected.
            
        Returns
        -------
            Torch Dataset. """
        if dataset == 'cora':
            if not undirected:
                # Load netset
                graph = load_netset(dataset)
                adjacency = graph.adjacency
                biadjacency = graph.biadjacency
                names = graph.names
                labels_true = graph.labels
                # torch.Data object
                data = Data(x=torch.FloatTensor(biadjacency.todense()),
                                edge_index=from_scipy_sparse_matrix(adjacency)[0],
                                y=torch.tensor(labels_true),
                                num_classes=len(np.unique(labels_true)))
            else:
                data = Planetoid(root='/tmp/Cora', name='Cora')
        elif dataset == 'citeseer':
            data = Planetoid(root='/tmp/Citeseer', name='Citeseer')
        elif dataset == 'pubmed':
            data = Planetoid(root='/tmp/Pubmed', name='Pubmed')
        return data
        

    def k_fold(self, data: Data, k: int, random_state: int) -> tuple:
        """Split all data in Data into k folds. Each fold contains train/val/test splits, where val and test sizes equal 1/k.
        
        Parameters
        ----------
        data: Data
            torch.Data wrapper containing graph and feature information.
        k: int
            k in k-folds method.
        random_state: int
            Controls the reproducility.
            
        Returns
        -------
            Tuple of train/val/test indices for each fold.
        """
        skf = StratifiedKFold(k, shuffle=True, random_state=random_state)

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
