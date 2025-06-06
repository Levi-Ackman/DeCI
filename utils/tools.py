import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


def adjust_learning_rate(optimizer, epoch,args):
    if args.lradj == 'binary':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
    elif args.lradj == 'type0':
        lr_adjust = {epoch: args.learning_rate if epoch<1 else args.learning_rate * (0.5 ** (((epoch-1)) // 1))}
    elif args.lradj == 'type05':
        lr_adjust = {epoch: args.learning_rate if epoch<5 else args.learning_rate * (0.5 ** (((epoch-4)) // 1))}
    elif args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate if epoch<10 else args.learning_rate * (0.5 ** (((epoch-9)) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate if epoch<20 else args.learning_rate * (0.5 ** (((epoch-19)) // 1))}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch<30 else args.learning_rate * (0.5 ** (((epoch-29)) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch<40 else args.learning_rate * (0.5 ** (((epoch-39)) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if args.print_process: print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def evaluate(labels, y_hats, classes=2,y_probs=None):
    accuracy = accuracy_score(labels, y_hats)
    precision = precision_score(labels, y_hats, average='macro', zero_division=0)
    recall = recall_score(labels, y_hats, average='macro', zero_division=0)
    f1 = f1_score(labels, y_hats, average='macro', zero_division=0)
    try:
        if classes > 2:
            if y_probs is not None:
                labels_one_hot = np.eye(classes)[labels.astype(int)]
                roc_auc = roc_auc_score(labels_one_hot, y_probs, multi_class='ovr', average='macro')
            else:
                labels_one_hot = np.eye(classes)[labels.astype(int)]
                y_hats_one_hot = np.eye(classes)[y_hats.astype(int)]
                roc_auc = roc_auc_score(labels_one_hot, y_hats_one_hot, multi_class='ovr', average='macro')
        else:
            if y_probs is not None:
                roc_auc = roc_auc_score(labels, y_probs)  
            else:
                roc_auc = roc_auc_score(labels, y_hats)  
    except ValueError:
        roc_auc = float('nan')  

    return accuracy, precision, recall, f1, roc_auc