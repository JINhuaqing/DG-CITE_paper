import torch.nn as nn
from torch.functional import F
# Generator Code

class Generator(nn.Module):
    def __init__(self, d, nz):
        super(Generator, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.layer_initx = nn.Linear(d, 64)
        self.layer_initz = nn.Linear(nz, 64)
        self.layer1 = nn.Sequential(
            nn.Linear(64, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.Linear(64, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(64, 1), 
        )

    def forward(self, x, z):
        x = F.relu(self.layer_initx(x))
        z = F.relu(self.layer_initz(z))
        x = x+z
        
        # res net 
        residual = x
        x = self.layer1(x)
        x = F.relu(x+residual)
        #x = self.dropout(x)
        
        x = self.layer2(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, d):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.layer_initx = nn.Linear(d, 64)
        self.layer_inity = nn.Linear(1, 64)
        self.layer1 = nn.Sequential(
            nn.Linear(64, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.Linear(64, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            #nn.Linear(128, 64), 
            #nn.BatchNorm1d(64),
            #nn.ReLU(), 
            #nn.Dropout(0.5),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = F.relu(self.layer_initx(x))
        y = F.relu(self.layer_inity(y))
        x = x+y
        
        # res net 
        residual = x
        x = self.layer1(x)
        x = F.relu(x+residual)
        x = self.dropout(x)
        
        x = self.layer2(x)
        return x