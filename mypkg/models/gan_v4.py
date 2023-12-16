# compared to gan_v2, it make script has output (linear_term, linear+noise)
import torch.nn as nn
from torch.functional import F
# Generator Code

class Generator(nn.Module):
    def __init__(self, d, nz):
        super(Generator, self).__init__()
        inner_nfs = 256
        self.dropout = nn.Dropout(0.5)
        self.layer_initx = nn.Linear(d, 64)
        self.layer_initz = nn.Linear(nz, 64)
        self.layer1 = nn.Sequential(
            nn.Linear(64, inner_nfs), 
            nn.BatchNorm1d(inner_nfs),
            nn.ReLU(), 
            nn.Linear(inner_nfs, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, inner_nfs), 
            nn.BatchNorm1d(inner_nfs),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(inner_nfs, 1), 
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, inner_nfs), 
            nn.BatchNorm1d(inner_nfs),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(inner_nfs, 1), 
        )

    def forward(self, x, z):
        x = F.relu(self.layer_initx(x))
        z = F.relu(self.layer_initz(z))
        y = x+z
        
        # res net 
        residual = y
        y = self.layer1(y)
        y = F.relu(y+residual)
        
        lin = self.layer2(x)
        noise = self.layer3(y)
        return lin, noise+lin
    
class Discriminator(nn.Module):
    def __init__(self, d):
        super(Discriminator, self).__init__()
        inner_nfs = 64
        self.dropout = nn.Dropout(0.5)
        self.layer_initx = nn.Linear(d, 64)
        self.layer_inity = nn.Linear(1, 64)
        self.layer1 = nn.Sequential(
            nn.Linear(64, inner_nfs), 
            nn.BatchNorm1d(inner_nfs),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(inner_nfs, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, inner_nfs), 
            nn.BatchNorm1d(inner_nfs),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(inner_nfs, 1), 
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
