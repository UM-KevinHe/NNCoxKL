from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class Data(Dataset):
    def __init__(self, X, y, delta_tilde=None, use_gpu=False):        
        sorted_indices = np.argsort(y[:, 1])
        X = X[sorted_indices]
        y = y[sorted_indices]

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        
        if delta_tilde is not None:
            self.delta_tilde = delta_tilde[sorted_indices]
            self.delta_tilde = torch.from_numpy(self.delta_tilde.astype(np.float32))
        else:
            self.delta_tilde = None
        
        if use_gpu and torch.cuda.is_available():
            self.X = self.X.cuda()
            self.y = self.y.cuda()
            if self.delta_tilde is not None:
                self.delta_tilde = self.delta_tilde.cuda()
        
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        if self.delta_tilde is not None:
            return self.X[index], self.y[index], self.delta_tilde[index]
        else:
            return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
    def subset(self, indices):
        X_subset = self.X[indices].cpu().numpy()
        y_subset = self.y[indices].cpu().numpy()
        
        if self.delta_tilde is not None:
            delta_tilde_subset = self.delta_tilde[indices].cpu().numpy()
            return Data(X_subset, y_subset, delta_tilde_subset)
        else:
            return Data(X_subset, y_subset)
    