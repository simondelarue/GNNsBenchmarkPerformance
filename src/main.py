import argparse
import numpy as np
import os

from dataset import PlanetoidDataset


if __name__=='__main__':
    DATAPATH = os.path.join(os.path.dirname(os.getcwd()), 'data')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--undirected', type=bool, required=True)
    parser.add_argument('--randomstate', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--logger', type=str, default=None)
    args = parser.parse_args()

    #raise Exception('end')

    # Create dataset
    dataset = PlanetoidDataset(dataset=args.dataset, random_state=args.randomstate,
                            k=args.k, undirected=args.undirected)
    
    cor = dataset.get_netset(args.dataset, DATAPATH, use_cache=True)
    
    print(f'Number of nodes: {dataset.data.x.shape[0]}')
    print(f'Number of edges: {len(dataset.data.edge_index[0])} (undirected: {args.undirected})')
    print(f'Number of classes: {dataset.data.num_classes}')

    # Get model
    model = args.model

    # Dictionary of arguments
    kwargs = {
        'dataset': dataset, 
        'undirected': args.undirected,
        'random_state': args.randomstate,
        'k': args.k,
        'model': model,
        'logger': args.logger 
    }