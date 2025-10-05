from data_provider.data_loader_CV import PPMI_Dataset,Mātai_Dataset,Neurocon_Dataset,Neurocon_Dataset,Abide_Dataset,ADNI_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

data_dict = {
    'PPMI': PPMI_Dataset,
    'Mātai': Mātai_Dataset,
    'Neurocon': Neurocon_Dataset,
    'Taowu': Neurocon_Dataset,
    'Abide': Abide_Dataset,
    'ADNI': ADNI_Dataset,
}

def collate_fn(batch, max_len):
    data, labels = zip(*batch)  
    padded_data = [x[:max_len] for x in data]  
    return torch.stack(padded_data), torch.tensor(labels)

def custom_collate_fn(batch, max_len):
    return collate_fn(batch, max_len=max_len)


def data_provider(args):
    Data = data_dict[args.data]
    kf = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    dataset = Data(args.data_path, args.data_type, args.protocol,args.seq_len)
    labels = dataset.labels
    drop_last=False
    
    unique_labels = np.unique(np.array(dataset.labels))
    num_categories = len(unique_labels)
    
    train_loaders=[]
    val_loaders=[]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.data, labels)):
        train_data = torch.utils.data.Subset(dataset, train_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=drop_last,shuffle=True, num_workers=args.num_workers,collate_fn=lambda x: custom_collate_fn(x, max_len=args.seq_len),)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=drop_last,shuffle=False, num_workers=args.num_workers,collate_fn=lambda x: custom_collate_fn(x, max_len=args.seq_len),)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        if args.print_data_info:
            
            train_labels = [train_data[i][1] for i in range(len(train_data))]
            train_samples_num=[]
            for i in range(num_categories):
                train_samples_num.append(train_labels.count(i))

            val_labels = [val_data[i][1] for i in range(len(val_data))]
            val_samples_num=[]
            for i in range(num_categories):
                val_samples_num.append(val_labels.count(i))

            print(f"Fold {fold + 1}:")
            print(f"  Training samples: {len(train_data)}")
            for i in range(num_categories):
                print(f'Number of Class {i} in training set: {train_samples_num[i]}')
            print(f"  Validation samples: {len(val_data)}")
            for i in range(num_categories):
                print(f'Number of Class {i} in validation set: {val_samples_num[i]}')
            
            sample_data, sample_label = next(iter(train_loader))
            print(f"Sample data shape: {sample_data.shape}, Sample label: {sample_label}")
    return train_loaders,val_loaders