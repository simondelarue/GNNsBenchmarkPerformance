import argparse
from collections import defaultdict
import numpy as np
import os
import torch
from utils import save_dict, check_exists
from dataset import PlanetoidDataset
from train import Trainer


def run(dataset, undirected, penalized, random_state, k, stratified, model):
    """Run experiment."""
    
    outs = defaultdict(list)

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*dataset.kfolds)):
        # Model training + predictions
        # In semi-supervised classification, model trains on the entire network (graph + features), but has access only to node labels in the training split. It is then evaluated on (val and) test split, in which labels are unknown.
        trainer = Trainer(train_idx, val_idx, test_idx)
        train_acc, test_acc, elapsed_time = trainer(model, dataset, penalized)

        #test_acc = Test(model, dataset, **kwargs)

        # Save training and test scores for averages on n runs
        outs['train acc'].append(train_acc)
        outs['test acc'].append(test_acc)
        outs['elapsed_time'].append(elapsed_time)

    print('Train acc: ', np.mean(outs['train acc']))
    print('Test acc: ', np.mean(outs['test acc']))
    return outs


if __name__=='__main__':
    DATAPATH = os.path.join(os.path.dirname(os.getcwd()), 'data')
    RUNPATH = os.path.join(os.path.dirname(os.getcwd()), 'runs')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--undirected', type=str, required=True)
    parser.add_argument('--penalized', type=str, required=True)
    parser.add_argument('--randomstate', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--stratified', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    #parser.add_argument('--logger', type=str, default=None)
    args = parser.parse_args()

    # Output filename
    filename = '_'.join([args.dataset,
                         'undir' + args.undirected.lower(),
                         'penalized' + args.penalized.lower(),
                         'random' + str(args.randomstate),
                         'k' + str(args.k),
                         'strat' + args.stratified.lower(),
                         args.model])

    # If run already exists, pass
    force_run = True
    check_exists(RUNPATH, filename, force_run=force_run)
    
    undirected = args.undirected.lower()=='true'
    penalized = args.penalized.lower()=='true'
    stratified = args.stratified.lower()=='true'

    # Create dataset for GNN based models
    dataset = PlanetoidDataset(dataset=args.dataset,
                               undirected=undirected,
                               random_state=args.randomstate, k=args.k,
                               stratified=stratified)
    
    # For baseline models
    netset_dataset = dataset.get_netset(args.dataset, DATAPATH, use_cache=True)
    
    print(f'Algorithm: {args.model}')
    print(f'Dataset: {args.dataset} (#nodes={dataset.data.x.shape[0]}, #edges={len(dataset.data.edge_index[0])}, undirected: {undirected})')
    print(f'kfolds: {args.k} (stratified: {stratified})')

    # Dictionary of arguments
    kwargs = {
        'dataset': dataset, 
        'undirected': undirected,
        'penalized': penalized,
        'random_state': args.randomstate,
        'k': args.k,
        'stratified': stratified,
        'model': args.model,
        #'logger': args.logger 
    }

    # Run experiment
    outs = run(**kwargs)

    # Save results
    global_out = {}
    global_out['meta'] = args
    global_out['results'] = outs
    save_dict(RUNPATH, filename, global_out)
    