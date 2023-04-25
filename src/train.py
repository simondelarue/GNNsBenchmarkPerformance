import torch

from utils import get_model


class Trainer:
    def __init__(self, train_idx, val_idx, test_idx):
        self.train_idx = train_idx
        self.val_idx = None # not used
        self.test_idx = torch.cat((val_idx, test_idx)) # concatenate val and test

    def __call__(self, model: str, dataset, penalized):
        return self.train_eval(model, dataset, penalized)

    def train_eval(self, model, dataset, penalized):
        train_acc = 0
        test_acc = 0
        for _ in range(3):
            alg = get_model(model, dataset)
            labels_pred = alg.fit_predict(dataset, self.train_idx, self.val_idx, self.test_idx)
            train_acc += alg.accuracy(dataset, labels_pred, self.train_idx, penalized, 'train')
            test_acc += alg.accuracy(dataset, labels_pred, self.test_idx, penalized, 'test')

        avg_train_acc = train_acc / 3
        avg_test_acc = test_acc / 3

        return (avg_train_acc, avg_test_acc)
        