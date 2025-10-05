import os
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

class PPMI_Dataset(Dataset):
    """
    A custom PyTorch dataset for loading time series data from PPMI (Parkinson's Progression Markers Initiative) dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=210
        Total 182 samples. (Importantly, we removed some outlier with length != 210 )
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/ppmi", data_type='TS',protocol="schaefer100",seq_len=210):
        self.source_dir = source_dir
        # 0: control 1:patient 2: prodromal 3:swedd
        self.categories = ['control', 'patient', 'prodromal', 'swedd']
        self.data = []
        self.labels = []
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        mat_data = scipy.io.loadmat(file_path)
                        self.data.append(torch.tensor(mat_data['data'], dtype=torch.float32))
                        self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class Mātai_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the Mātai dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=200
        Total 60 samples.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/Mātai_dataset", data_type='TS',protocol="schaefer100",seq_len=200):
        self.source_dir = source_dir
        # 0:baseline 1:postseason
        self.categories = ['baseline', 'postseason']
        self.data = []
        self.labels = []
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        mat_data = scipy.io.loadmat(file_path)
                        self.data.append(torch.tensor(mat_data['data'], dtype=torch.float32))
                        self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class Neurocon_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the Neurocon dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=137
        Total 41 samples.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/neurocon", data_type='TS',protocol="schaefer100",seq_len=137):
        self.source_dir = source_dir
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        self.data = []
        self.labels = []
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        mat_data = scipy.io.loadmat(file_path)
                        self.data.append(torch.tensor(mat_data['data'], dtype=torch.float32))
                        self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class Taowu_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the Taowu dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=239
        Total 40 samples.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/taowu", data_type='TS',protocol="schaefer100",seq_len=239):
        self.source_dir = source_dir
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        self.data = []
        self.labels = []
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        mat_data = scipy.io.loadmat(file_path)
                        self.data.append(torch.tensor(mat_data['data'], dtype=torch.float32))
                        self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Abide_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the ABIDE dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=120-300 (Importantly, the original length of the ABIDE time series range from 120 to 300. 
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/abide", data_type='TS',protocol="ward100",seq_len=300):
        self.source_dir = source_dir
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        self.data = []
        self.labels = []
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.uniform_length = seq_len
        self.load_data()
    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        mat_data = scipy.io.loadmat(file_path)
                        signal = torch.tensor(mat_data['data'], dtype=torch.float32)
                        original_length = signal.shape[0]
                        if original_length != self.uniform_length:
                            pass
                        else:
                            signal_resized = signal
                            self.data.append(signal_resized)
                            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class ADNI_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the ADNI dataset.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/ADNI/ADNI", data_type='TS', protocol="AAL116", seq_len=197):
        self.source_dir = source_dir
        self.categories = ['Control', 'MCI', 'AD']   # 0,1,2
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            for fname in os.listdir(category_dir):
                fpath = os.path.join(category_dir, fname)
                arr = np.load(fpath)
                x = torch.from_numpy(arr.astype(np.float32)) 
                y = torch.tensor(label, dtype=torch.long)
                self.data.append(x)
                self.labels.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
