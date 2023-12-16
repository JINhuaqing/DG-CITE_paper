#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


from constants import RES_ROOT, FIG_ROOT, DATA_ROOT
from utils.misc import load_pkl, save_pkl, merge_intervals, moving_average
from utils.colors import qual_cmap
from utils.stats import weighted_quantile
from data_gen import get_simu_data
from torch.utils.data import DataLoader
from gan.utils import MyDataSet
from models.ddpm import ContextNet, ddpm_schedules, DDPM
from ddpm.train_ddpm import TrainDDPM


# In[4]:


import torch
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
from tqdm import tqdm, trange
import random
from joblib import Parallel, delayed


# In[5]:


from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np


# In[6]:


torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)


# In[7]:


plt.style.use(FIG_ROOT/"base.mplstyle")


# In[8]:


n = 500
d = 10
rho = 0.9
is_homo = True
dftype = torch.get_default_dtype();

data = get_simu_data(n=n, d=d, rho=rho, is_homo=is_homo);
data_test = get_simu_data(n=1000, d=d, rho=rho, is_homo=is_homo);

data_train = MyDataSet(Y=data.Y1, X=data.X);


# In[9]:


myddpm = TrainDDPM(data_train, save_dir="test", verbose=True, device="cuda");


# In[ ]:


myddpm.train(n_epoch=10000, save_snapshot=True)


# In[218]:

