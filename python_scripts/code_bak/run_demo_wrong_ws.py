#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


from constants import RES_ROOT, FIG_ROOT, DATA_ROOT
from utils.misc import load_pkl, save_pkl, merge_intervals
from utils.colors import qual_cmap
from utils.stats import weighted_quantile
from data_gen import get_simu_data
from models.gan_v2 import Generator, Discriminator
from gan.gan_train import MyGAN
from gan.utils import MyDataSet
from demo_settings import simu_settings
from CQR import get_CQR_CIs


# In[3]:



import torch
import scipy.stats as ss
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as ddict
from tqdm import tqdm, trange
import random
from joblib import Parallel, delayed
import pandas as pd
from pprint import pprint


# In[ ]:
import argparse
parser = argparse.ArgumentParser(description='run demo')
parser.add_argument('-s', '--setting', type=str, help='the simu setting') 
args = parser.parse_args()





# # Params

# In[5]:


setting = args.setting


# In[6]:


params = edict()

params.simu_setting = edict()
params.simu_setting.rho = 0.9
params.simu_setting.is_homo = False
params.simu_setting.n = 1000
params.simu_setting.d = 100
params.simu_setting.ntest = 10000
params.simu_setting.cal_ratio = 0.25
params.simu_setting.update(simu_settings[setting])
pprint(params.simu_setting)


params.nrep = 100 # num of reptition for simulation
params.K = 40 # num of sps drawn from q(Y(1)|X)
params.nz = 5 # the dim of features for GAN
params.save_snapshot = True

params.gan_training = edict()
# Batch size during training
params.gan_training.batch_size = 32
# Number of training epochs
params.gan_training.num_epochs = 500
# Learning rate for optimizers
params.gan_training.lr = 0.002
# Beta1 hyperparameter for Adam optimizers
params.gan_training.beta1 = 0.5
params.gan_training.device="cpu"


params.hypo_test = edict()
params.hypo_test.alpha = 0.05 # sig level

params.prefix = ""
params.save_dir = f"demo_{setting}0"
if not (RES_ROOT/params.save_dir).exists():
    (RES_ROOT/params.save_dir).mkdir()


# In[ ]:





# In[ ]:





# # Some fns

# In[7]:


def _gen_Y_given_X(X, netG):
    """With generative model f(Y|X), generate Y given X
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_mul = torch.tensor(X * np.ones((params.K, 1)));
    Zs = torch.randn(params.K, params.nz);
    with torch.no_grad():
        Yhat = netG(X_mul, Zs).view(-1).numpy()
    return Yhat

def _get_pred_intv(X, qv, netG):
    teY_hat = _gen_Y_given_X(X, netG);
    intvs = np.stack([teY_hat-qv, teY_hat+qv]).T
    return merge_intervals(intvs)

def _get_metric(v, intvs):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    in_sets = np.sum([np.bitwise_and(v>intv[0], v<intv[1]) for intv in intvs], axis=0)
    intvs_len = np.sum([np.diff(intv) for intv in intvs])
    metrics = edict()
    metrics.in_sets = in_sets
    metrics.intvs_len = intvs_len
    return metrics


# In[ ]:





# # Simu

# In[8]:


def _run_fn(rep_ix):
    manualSeed = rep_ix
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)

    data_train = get_simu_data(n=params.simu_setting.n, 
                               d=params.simu_setting.d, 
                               is_homo=params.simu_setting.is_homo, 
                               rho=params.simu_setting.rho);
    data_test = get_simu_data(n=params.simu_setting.ntest, 
                               d=params.simu_setting.d, 
                               is_homo=params.simu_setting.is_homo, 
                               rho=params.simu_setting.rho);
    n_train = int(data_train.T.sum()*(1-params.simu_setting.cal_ratio));
    tr_idxs = np.sort(np.random.choice(np.where(data_train.T)[0], size=n_train, replace=False));
    cal_idxs = np.setdiff1d(np.where(data_train.T)[0], tr_idxs);
    
    
    # train q(Y(1)|X)
    # I skip this for now, suppose you get one
    data_train_gan = MyDataSet(Y=data_train.Y[tr_idxs], X=data_train.X[tr_idxs])
    input_params = params.gan_training.copy()
    input_params.pop("num_epochs")
    my_gan =  MyGAN(Generator=Generator, Discriminator=Discriminator, nz=params.nz,
                    d=params.simu_setting.d, train_data=data_train_gan, save_dir=params.save_dir, verbose=False,
                    prefix=f"rep{rep_ix}",
                    **input_params);
    my_gan.train(num_epochs=params.gan_training.num_epochs, data_test=None, save_snapshot=params.save_snapshot)
    netG = my_gan.netG;
    #netG = my_gan.get_opt_model();
    netG.eval();
    
    
    # propensity score
    # I can use all data in train set (including cal set)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0)
    psY = data_train.T.astype(int)
    psX = data_train.X
    clf = clf.fit(psX, psY);
    def psfun(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return clf.predict_proba(x)[:, 1]
    
    
    # weight function
    def wtfun(x):
        # for unconditional weight
        return 1/psfun(x)
    
    
    # get the weight and nonconformity score for each data point in cal set
    wE_set = []
    for cal_idx in cal_idxs:
        cal_X = data_train.X[cal_idx]
        cal_Y = data_train.Y1[cal_idx]
        w = wtfun(cal_X)
        
        # draw K sps from q(Y|X)
        Yhat = _gen_Y_given_X(cal_X, netG)
            
        # the nonconformity score
        E = np.abs(Yhat - cal_Y).min()
        
        wE_set.append([w[0], E])
        
    # now we get weighted quantile
    ws = np.array(wE_set)[:, 0]
    Es = np.array(wE_set)[:, 1];
    qv = weighted_quantile(Es, [(1-params.hypo_test.alpha)*(1/len(Es)+1)], sample_weight=ws);
    
    
    # evaluate on test data
    
    test_res = []
    for te_idx in range(params.simu_setting.ntest):
        teX = data_test.X[te_idx]
        teY1 = data_test.Y1[te_idx]
        tetau = data_test.tau[te_idx]
        intvs = _get_pred_intv(teX, qv, netG)
        test_res.append(_get_metric([teY1, tetau], intvs))
    prbs = np.mean([res['in_sets'] for res in test_res], axis=0)
    mlen = np.mean([res['intvs_len'] for res in test_res])
    
    
    # results from CQR
    CQR_CIs = get_CQR_CIs(X=data_train.X, Y=data_train.Y, 
                          T=data_train.T, Xtest=data_test.X, 
                          nav=0, 
                          alpha=params.hypo_test.alpha, 
                          fyx_est="quantBoosting", seed=manualSeed)
    mlen_cqr = np.mean(CQR_CIs[:, 1] -  CQR_CIs[:, 0])
    prb_Y1_cqr = np.bitwise_and(data_test.Y1>CQR_CIs[:, 0], data_test.Y1<CQR_CIs[:, 1]).mean()
    prb_tau_cqr = np.bitwise_and(data_test.tau>CQR_CIs[:, 0], data_test.tau<CQR_CIs[:, 1]).mean()
    
    res_all = edict()
    res_all.GAN = (prbs, mlen)
    res_all.CQR = ([prb_Y1_cqr, prb_tau_cqr], mlen_cqr)
    return res_all


# In[9]:


with Parallel(n_jobs=30) as parallel:
    test_ress = parallel(delayed(_run_fn)(rep_ix) for rep_ix in tqdm(range(params.nrep), total=params.nrep))

save_pkl((RES_ROOT/params.save_dir)/f"res_demo.pkl", test_ress, is_force=True)


# In[10]:
