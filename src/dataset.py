import numpy as np
import os
import pickle
from scipy import sparse

from base import BaseDataset

from sknetwork.data import load_netset, Bunch

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class WikivitalsDataset(BaseDataset):
    """Wikivitals networks: Wikivitals, Wikivitals-fr, Wikischools."""

    def __init__(self, dataset: str, undirected: bool, random_state: int, k: int, stratified: bool):
        super(WikivitalsDataset, self).__init__(dataset, undirected, random_state, k, stratified)
    
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
            print(f'Load netset data...')
            graph_raw = load_netset(dataset)
            
            graph = Bunch()
            graph.adjacency = graph_raw.adjacency
            graph.biadjacency = graph_raw.biadjacency
            graph.names = graph_raw.names
            graph.labels_true = graph_raw.labels

            # Save Netset dataset
            with open(os.path.join(pathname, dataset), 'bw') as f:
                pickle.dump(graph, f)
            print(f'Netset data saved in {os.path.join(pathname, dataset)}')
        
        self.netset = graph
        
        return self.netset
    
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
            raise Exception('Wikivitals dataset should be used with argument --undirected=true.')
        
        return data
    
    
class PlanetoidDataset(BaseDataset):
    """Citation networks: Cora, PubMed, Citeseer."""

    def __init__(self, dataset: str, undirected: bool, random_state: int, k: int, stratified: bool):
        super(PlanetoidDataset, self).__init__(dataset, undirected, random_state, k, stratified)

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
        n = len(self.data.y) #len(set(rows).union(set(cols)))
        adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        # Features
        biadjacency = sparse.csr_matrix(np.array(self.data.x), dtype=bool)

        # Node information
        labels = np.array(self.data.y)

        graph = Bunch()
        graph.adjacency = adjacency
        graph.biadjacency = biadjacency
        graph.labels_true = labels

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
                data = self._to_custom_data(Planetoid(root=f'/tmp/{dataset.capitalize()}',
                                                      name=f'{dataset.capitalize()}'))
        elif dataset in ['pubmed', 'citeseer']:
            data = self._to_custom_data(Planetoid(root=f'/tmp/{dataset.capitalize()}',
                                                      name=f'{dataset.capitalize()}'))
        elif dataset in ['reddit']:
            data = self._to_custom_data(Reddit(root=f'tmp/{dataset.capitalize()}'))
        return data
    
    def _to_custom_data(self, dataset):
        """Convert Dataset format from Pytorch to a modifiable Data object."""
        data = Data(x=dataset.x,
               edge_index=dataset.edge_index,
               num_classes=dataset.num_classes,
               y=dataset.y,
               train_mask=dataset.train_mask,
               val_mask=dataset.val_mask,
               test_mask=dataset.test_mask)
        
        return data
