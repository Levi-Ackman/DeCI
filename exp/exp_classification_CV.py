import os
import time
import warnings
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping,evaluate,adjust_learning_rate
from data_provider.data_factory_CV import data_provider
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax  

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        self.initial_model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.initial_model = nn.DataParallel(self.initial_model, device_ids=self.args.device_ids)
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in self.model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

        return self.model, self.initial_model
    def reset_model(self):
        self.model.load_state_dict(self.initial_model.state_dict())
        if self. args.print_process: print("Model has been reset to initial weights.")

    def _get_data(self):
        train_loaders, val_loaders= data_provider(self.args)
        return train_loaders, val_loaders
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def _select_criterion(self):
        criterion = nn.MSELoss() if (self.args.loss=='MSE' or self.args.loss=='mse') else nn.CrossEntropyLoss()
        return criterion
    
    def val(self, val_loader, criterion):
        total_loss = []
        self.model.eval()
        preds=[]
        targets=[]
        probs=[]
        with torch.no_grad():
            for _, (x_enc,label) in enumerate(val_loader):
                x_enc=x_enc.to(self.device)
                label=label.to(self.device)
                
                if self.args.classes!=2:
                    one_hot_label = torch.zeros(len(label), self.args.classes).to(self.device)
                    one_hot_label.scatter_(1, label.unsqueeze(1), 1)
                    label=one_hot_label
                
                y_hat = self.model(x_enc)
                loss = criterion(y_hat, label)   
                total_loss.append(loss.cpu().numpy())
                
                if self.args.classes!=2:
                    prob = torch.nn.functional.softmax(y_hat)
                    pred = torch.argmax(prob, dim=1).cpu().numpy()
                    target=torch.argmax(label, dim=1).cpu().numpy()
                else:
                    prob=y_hat.squeeze(-1)
                    pred = (prob > 0.5).to(label.dtype).cpu().numpy()
                    target = label.cpu().numpy()
                probs.append(prob.detach().cpu().numpy())
                preds.append(pred)
                targets.append(target)
            if len(preds)>0:
                preds = np.concatenate(preds, axis=0)    
                targets = np.concatenate(targets, axis=0)  
                probs = np.concatenate(probs, axis=0)  
            else:
                preds=preds[0]
                targets=targets[0]
                probs=probs[0]
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss,evaluate(targets, preds,self.args.classes,probs)

    def train(self, train_loader, val_loader,check_path):
        if not os.path.exists(check_path):
            os.makedirs(check_path)
        
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=self. args.print_process)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            train_loss = []
            preds=[]
            targets=[]
            probs=[]
            self.model.train()
            epoch_time = time.time()
            for _, (x_enc,label) in enumerate(train_loader):
                x_enc=x_enc.to(self.device)
                label=label.to(self.device)
                
                if self.args.classes!=2:
                    one_hot_label = torch.zeros(len(label), self.args.classes).to(self.device)
                    one_hot_label.scatter_(1, label.unsqueeze(1), 1)
                    label=one_hot_label
                else:
                    label = label.to(torch.float32).to(self.device) # convert to float32 for mse loss
                
                
                y_hat= self.model(x_enc)
                loss = criterion(y_hat, label)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.cpu().detach().numpy())
                if self.args.classes!=2:
                    prob = torch.nn.functional.softmax(y_hat)
                    pred = torch.argmax(prob, dim=1).cpu().numpy()
                    target=torch.argmax(label, dim=1).cpu().numpy()
                else:
                    prob=y_hat.squeeze(-1)
                    pred = (prob > 0.5).to(label.dtype).cpu().numpy()
                    target = label.cpu().numpy()
                
                probs.append(prob.detach().cpu().numpy())
                preds.append(pred)
                targets.append(target)
            if len(preds)>0:
                preds = np.concatenate(preds, axis=0)    
                targets = np.concatenate(targets, axis=0)   
                probs = np.concatenate(probs, axis=0) 
            else:
                preds=preds[0]
                targets=targets[0]
                probs=probs[0]
            train_metric =evaluate(targets, preds,self.args.classes)
            train_loss = np.average(train_loss)
            val_loss,val_metric = self.val(val_loader,criterion)

            if self. args.print_process:
                print(f"\n Epoch: {epoch + 1}, | Train Loss: {train_loss:.4f} Train Acc: {train_metric[0]:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_metric[0]:.4f}")
            early_stopping(-val_metric[0], self.model, check_path)
            if self. args.print_process:
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        return 

    def kf_train(self, setting):
        train_loaders,val_loaders=self._get_data()
        val_metrics=[]
        for fold, (train_loader, val_loader) in tqdm(enumerate(zip(train_loaders, val_loaders)), total=self.args.kfold, desc="Cross-validation", ncols=100):
            check_path = os.path.join(self.args.checkpoints, setting+'fold'+str(fold + 1))
            self.reset_model()
            if self. args.print_process: print(f"Fold {fold + 1}/{self.args.kfold} Start>>>>>>>>>>>>>>>>>>>\n")
            self.train(train_loader, val_loader, check_path)
            
            self.model.load_state_dict(torch.load(os.path.join(check_path, 'checkpoint.pth')))
            _,val_metric = self.val(val_loader,self._select_criterion())
            val_metrics.append(val_metric)
            
            ## Uncomment below code for save space on device
            if self. args.del_weight: 
                self.del_weight(check_path)
            if self. args.print_process: print(f"Fold {fold + 1}/{self.args.kfold} End<<<<<<<<<<<<<<<<<<<\n")
        avg_metric=[np.mean([val_metrics[i][j] for i in range(self.args.kfold)]) for j in range(5)]
        print(f'Total Avg accuracy: {avg_metric[0]:.4f}, precision: {avg_metric[1]:.4f}, recall: {avg_metric[2]:.4f}, macro_f1: {avg_metric[3]:.4f}, roc_auc: {avg_metric[4]:.4f}')
        return avg_metric
    
    def del_weight(self, path):
        if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
            os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
            if self. args.print_process: print('Model weights deleted....')

    def svm(self, train_loader, val_loader):
        X_train, y_train = map(lambda batches: np.concatenate(batches, axis=0), zip(*train_loader))
        X_val,   y_val   = map(lambda batches: np.concatenate(batches, axis=0), zip(*val_loader))

        # Compute the Pearson correlation matrix (FC) for each sample and stack into shape (B, N, N)
        FC_train = np.stack([np.corrcoef(x.T) for x in X_train], axis=0)
        FC_val   = np.stack([np.corrcoef(x.T) for x in X_val],   axis=0)
        FC_train = np.nan_to_num(FC_train, nan=0.0)
        FC_val   = np.nan_to_num(FC_val,   nan=0.0)

        # Determine the number of ROIs (N) and create a boolean mask for the upper triangle (including diagonal)
        N = FC_train.shape[-1]
        mask = np.triu(np.ones((N, N), dtype=bool), k=0)  # shape = (N, N)

        # Extract and flatten only the upper-triangle (including diagonal) elements: shape -> (B, N*(N+1)/2)
        X_train_feat = FC_train[:, mask]
        X_val_feat   = FC_val[:,   mask]

        # Initialize and train the SVM with probability outputs enabled
        svm = SVC(C=0.1, kernel='rbf', probability=True, random_state=self.args.seed)
        svm.fit(X_train_feat, y_train)

        # Predict class probabilities
        proba = svm.predict_proba(X_val_feat)
        if self.args.classes == 2:
            # For binary classification, take the probability of the positive class
            probs = proba[:, 1]
            preds = (probs > 0.5).astype(int)
        else:
            # For multiclass, apply softmax (optional calibration) and choose the class with highest probability
            probs = softmax(proba, axis=1)
            preds = np.argmax(probs, axis=1)

        # Evaluate predictions against ground truth
        val_metric = evaluate(y_val, preds, self.args.classes, probs)
        return None, val_metric

    def rf(self, train_loader, val_loader):
        # Concatenate all batches of time-series data (B, T, N) and labels
        X_train_ts, y_train = map(lambda batches: np.concatenate(batches, axis=0), zip(*train_loader))
        X_val_ts,   y_val   = map(lambda batches: np.concatenate(batches, axis=0), zip(*val_loader))

        # Compute the Pearson correlation matrix (FC) for each sample and stack into shape (B, N, N)
        FC_train = np.stack([np.corrcoef(x.T) for x in X_train_ts], axis=0)
        FC_val   = np.stack([np.corrcoef(x.T) for x in X_val_ts],   axis=0)
        FC_train = np.nan_to_num(FC_train, nan=0.0)
        FC_val   = np.nan_to_num(FC_val,   nan=0.0)

        # Determine the number of ROIs (N) and create a boolean mask for the upper triangle (including diagonal)
        N    = FC_train.shape[-1]
        mask = np.triu(np.ones((N, N), dtype=bool), k=0)  # shape = (N, N)

        # Extract and flatten only the upper-triangle (including diagonal) elements: shape -> (B, N*(N+1)/2)
        X_train_feat = FC_train[:, mask]
        X_val_feat   = FC_val[:,   mask]

        # Initialize and train the Random Forest classifier with specified hyperparameters
        rf_clf = RandomForestClassifier(
            n_estimators=12,  # number of trees
            max_depth=2,        # maximum tree depth
            random_state=self.args.seed,
            n_jobs=4               # number of parallel jobs
        )
        rf_clf.fit(X_train_feat, y_train)

        # Predict class probabilities
        proba = rf_clf.predict_proba(X_val_feat)
        if self.args.classes == 2:
            # For binary classification, take the probability of the positive class
            probs = proba[:, 1]
            preds = (probs > 0.5).astype(int)
        else:
            # For multiclass, apply softmax (optional calibration) and choose the class with highest probability
            probs = softmax(proba, axis=1)
            preds = np.argmax(probs, axis=1)

        # Evaluate predictions against ground truth
        val_metric = evaluate(y_val, preds, self.args.classes, probs)
        return None, val_metric


    
    def kf_ML(self, setting):
        train_loaders,val_loaders=self._get_data()
        val_metrics=[]
        for fold, (train_loader, val_loader) in tqdm(enumerate(zip(train_loaders, val_loaders)), total=self.args.kfold, desc="Cross-validation", ncols=100):
            _,val_metric = self.svm(train_loader, val_loader) if (self.args.Method == 'SVM' or self.args.Method=='svm') else self.rf(train_loader, val_loader)
            val_metrics.append(val_metric)
        avg_metric=[np.mean([val_metrics[i][j] for i in range(self.args.kfold)]) for j in range(5)]
        print(f'Total Avg accuracy: {avg_metric[0]:.4f}, precision: {avg_metric[1]:.4f}, recall: {avg_metric[2]:.4f}, macro_f1: {avg_metric[3]:.4f}, roc_auc: {avg_metric[4]:.4f}')
        return avg_metric