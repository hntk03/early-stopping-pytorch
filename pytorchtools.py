import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_accuracy = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_accuracy_max = 0.0

    def __call__(self, val_loss, val_accuracy,model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_accuracy = val_accuracy
            self.save_checkpoint(val_loss, val_accuracy, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_accuracy = val_accuracy
            self.save_checkpoint(val_loss, val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_accuracy, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased (loss {self.val_loss_min:.6f} --> {val_loss:.6f} acc {self.val_accuracy_max:.6f} --> {val_accuracy:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        self.val_accuracy_max = val_accuracy
