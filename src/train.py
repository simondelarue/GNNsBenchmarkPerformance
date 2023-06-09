import torch
import time

from utils import get_model


class Trainer:
    """Trainer class."""
    def __init__(self, train_idx, val_idx, test_idx):
        self.train_idx = train_idx
        self.val_idx = None # not used
        self.test_idx = torch.sort(torch.cat((val_idx, test_idx))).values # concatenate val and test
        self.N_RUNS = 3

    def __call__(self, model: str, dataset, penalized, **kwargs):
        return self.train_eval(model, dataset, penalized, **kwargs)

    def train_eval(self, model, dataset, penalized, **kwargs):
        """Train algorithm on several runs and compute average performance on train and test spits.
        
        Parameters
        ----------
        model: str
            Model name.
        dataset
            Custom Dataset object
        penalized: bool
            If true, labels not predicted (with value -1) are considered in the accuracy computation.
            
        Returns
        -------
            Tuple of train and test average accuracies on several runs, with corresponding running time.
        """
        train_acc = 0
        test_acc = 0
        elapsed_time = 0
        
        for _ in range(self.N_RUNS):
            # Get model
            alg = get_model(model, dataset, self.train_idx, **kwargs)
            
            # Training algorithm
            start = time.time()
            labels_pred = alg.fit_predict(dataset, self.train_idx, self.val_idx, self.test_idx, **kwargs)
            end = time.time()
            
            # Accuracy
            train_acc += alg.accuracy(dataset, labels_pred, self.train_idx, penalized, 'train')
            test_acc += alg.accuracy(dataset, labels_pred, self.test_idx, penalized, 'test')
            #print(f"Fold train acc: {alg.accuracy(dataset, labels_pred, self.train_idx, penalized, 'train')}")
            #print(f"Fold test acc: {alg.accuracy(dataset, labels_pred, self.test_idx, penalized, 'test')}")
            elapsed_time += end - start
            #print(f"Test acc: {alg.accuracy(dataset, labels_pred, self.test_idx, penalized, 'test')}")

        avg_train_acc = train_acc / self.N_RUNS
        avg_test_acc = test_acc / self.N_RUNS
        avg_elapsed_time = elapsed_time / self.N_RUNS

        return (avg_train_acc, avg_test_acc, avg_elapsed_time)
        