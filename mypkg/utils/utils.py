from torch.utils.data import Dataset
import numpy as np
class MyDataSet(Dataset):
    def __init__(self, Y, X):
        
        assert Y.shape[0] == X.shape[0], "X and Y should have same length"
        self.Y = Y
        self.X = X
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx]
        return x, y
    
    

