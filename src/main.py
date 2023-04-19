import argparse
import os

from baseline import Baseline
from utils import get_model
from dataset import PlanetoidDataset


def run(dataset, undirected, random_state, k, model, logger):
    """Run experiment."""
    
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*dataset.kfolds)):
        print(fold, len(train_idx), len(test_idx), len(val_idx))

        # Model training
        labels_pred = model.fit_predict(dataset, train_idx)
        print(len(labels_pred))


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
    
    netset_dataset = dataset.get_netset(args.dataset, DATAPATH, use_cache=True)
    
    print(f'Number of nodes: {dataset.data.x.shape[0]}')
    print(f'Number of edges: {len(dataset.data.edge_index[0])} (undirected: {args.undirected})')
    print(f'Number of classes: {dataset.data.num_classes}')

    # Get model
    model = get_model(args.model)
    print('Model name: ', model.name)

    # Dictionary of arguments
    kwargs = {
        'dataset': dataset, 
        'undirected': args.undirected,
        'random_state': args.randomstate,
        'k': args.k,
        'model': model,
        'logger': args.logger 
    }

    # Run experiment
    run(**kwargs)