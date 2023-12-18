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
from utils.utils import MyDataSet, get_idx_sets
from demo_settings import simu_settings
from CQR import get_CQR_CIs
from models.ddpm import ContextNet, ddpm_schedules, DDPM
from ddpm.train_ddpm import TrainDDPM


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

import argparse
parser = argparse.ArgumentParser(description='run demo')
parser.add_argument('-s', '--setting', type=str, help='the simu setting') 
args = parser.parse_args()

# In[ ]:



setting = args.setting


params = edict()

params.simu_setting = edict()
params.simu_setting.rho = 0.9
params.simu_setting.is_homo = False
params.simu_setting.n = 1000
params.simu_setting.d = 10
params.simu_setting.ntest = 1000
params.simu_setting.cal_ratio = 0.25 # for conformal inference
params.simu_setting.val_ratio = 0.15 # for tuning network
params.simu_setting.update(simu_settings[setting])
params.simu_setting.n = 2000
pprint(params.simu_setting)


params.nrep = 100 # num of reptition for simulation
params.K = 40 # num of sps drawn from q(Y(1)|X)
params.save_snapshot = 100
params.dftype = torch.float32
params.device="cpu"
params.n_jobs = 30
params.verbose = False

params.ddpm_training = edict()
# Batch size during training
params.ddpm_training.batch_size = 128 
# Number of training epochs
params.ddpm_training.n_epoch = 1500
params.ddpm_training.n_infeat = 128
# Learning rate for optimizers
params.ddpm_training.lr = 0.002
params.ddpm_training.lr_gamma = 0.5
params.ddpm_training.test_intv = 5
params.ddpm_training.n_T = 100 # 100
params.ddpm_training.drop_prob = 0.1

params.wconformal = edict()
# remove too large and too small in ws/mean(ws)
params.wconformal.nwthigh = 20
params.wconformal.nwtlow = 0.05
params.wconformal.useinf = False


params.hypo_test = edict()
params.hypo_test.alpha = 0.05 # sig level

params.prefix = ""
params.save_dir = f"demo_ddmp_{setting}_sc1a"
if not (RES_ROOT/params.save_dir).exists():
    (RES_ROOT/params.save_dir).mkdir()


# In[ ]:





# # Some fns

# In[14]:


def _gen_Y_given_X(X, ddpm):
    """With generative model f(Y|X), generate Y given X
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    c = torch.tensor(X * np.ones((params.K, 1)), dtype=params.dftype).to(params.device);
    with torch.no_grad():
        x_0, _ = ddpm.sample(c, device=params.device, guide_w=0, is_store=False);
    x_0 = x_0.cpu().numpy().reshape(-1)
    return x_0

def _get_pred_intv(X, qv, ddpm):
    teY_hat = _gen_Y_given_X(X, ddpm);
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


# # Simu

# In[15]:


def _run_fn(rep_ix):
    manualSeed = rep_ix
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    torch.set_default_dtype(params.dftype)
    torch.set_default_device(params.device)
    data_train = get_simu_data(n=params.simu_setting.n, 
                               d=params.simu_setting.d, 
                               is_homo=params.simu_setting.is_homo, 
                               rho=params.simu_setting.rho);
    data_test = get_simu_data(n=params.simu_setting.ntest, 
                               d=params.simu_setting.d, 
                               is_homo=params.simu_setting.is_homo, 
                               rho=params.simu_setting.rho);
    
    cal_idxs, val_idxs, tr_idxs = get_idx_sets(all_idxs=np.where(data_train.T)[0], 
                                               ratios = [params.simu_setting.cal_ratio, params.simu_setting.val_ratio])
        
        
    # train q(Y(1)|X)
    # I skip this for now, suppose you get one
    data_train_ddpm = MyDataSet(Y=data_train.Y[tr_idxs], X=data_train.X[tr_idxs])
    data_val = edict()
    data_val.c =  data_train.X[val_idxs]
    data_val.x =  data_train.Y[val_idxs]
    
    input_params = params.ddpm_training.copy()
    input_params.pop("n_epoch")
    myddpm = TrainDDPM(data_train_ddpm, save_dir=params.save_dir, verbose=params.verbose, prefix=f"rep{rep_ix}", 
                       device=params.device,
                       **input_params);
    myddpm.train(n_epoch=params.ddpm_training.n_epoch, 
                 data_val=data_val, save_snapshot=params.save_snapshot)
    ddpm = myddpm.get_opt_model()
    #ddpm = myddpm.get_model(800)
    ddpm.eval()
        
    ## propensity score
    ## I can use all data in train set (including cal set)
    #from sklearn.linear_model import LogisticRegression
    #clf = LogisticRegression(random_state=0)
    #psY = data_train.T.astype(int)
    #psX = data_train.X
    #clf = clf.fit(psX, psY);
    #def psfun(x):
    #    if x.ndim == 1:
    #        x = x.reshape(1, -1)
    #    return clf.predict_proba(x)[:, 1]
    #
    #
    ## weight function
    #def wtfun(x):
    #    # for unconditional weight
    #    return 1/psfun(x)
    
    # weight function
    def wtfun(x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # for unconditional weight
        return np.ones(x.shape[0])
    
    # get the weight and nonconformity score for each data point in cal set
    def _run_fn1(cal_idx):
        torch.set_default_dtype(params.dftype)
        cal_X = data_train.X[cal_idx]
        cal_Y = data_train.Y1[cal_idx]
        w = wtfun(cal_X)
        
        # draw K sps from q(Y|X)
        Yhat = _gen_Y_given_X(cal_X, ddpm)
            
        # the nonconformity score
        E = np.abs(Yhat - cal_Y).min()
        
        return [w[0], E]
    if params.verbose:
        pbar1 = tqdm(cal_idxs, total=len(cal_idxs))
    else:
        pbar1 = cal_idxs
    with Parallel(n_jobs=params.n_jobs) as parallel:
        wE_set = parallel(delayed(_run_fn1)(cal_idx) for cal_idx in pbar1)
    ws = np.array(wE_set)[:, 0]
    Es = np.array(wE_set)[:, 1];
        
        
    # evaluate on test data
    
    def _run_fn2(te_idx):
        torch.set_default_dtype(params.dftype)
        teX = data_test.X[te_idx]
        teY1 = data_test.Y1[te_idx]
        tetau = data_test.tau[te_idx]
        
        # get qv for current test pt
        ws_wtest = np.concatenate([ws, wtfun(teX)]);
        Es_winf = np.concatenate([Es, [np.inf]]);
        nws_wtest = ws_wtest/ws_wtest.mean();
        nws_wtest[nws_wtest<params.wconformal.nwtlow] = params.wconformal.nwtlow
        nws_wtest[nws_wtest>params.wconformal.nwthigh] = params.wconformal.nwthigh
        # can not take the inf into consider
        #qv_cur = weighted_quantile(Es_winf, [1-params.hypo_test.alpha], sample_weight=nws_wtest);
        qt = (1 + nws_wtest[-1]/ nws_wtest[:-1].sum()) * (1 - params.hypo_test.alpha)
        if qt <= 1:
            qv_cur = weighted_quantile(Es, [qt], sample_weight=ws_wtest[:-1]);
        else:
            if params.wconformal.useinf:
                qv_cur = np.inf
            else:
                qv_cur = np.max(Es)
        
        intvs = _get_pred_intv(teX, qv_cur, ddpm)
        return _get_metric([teY1, tetau], intvs)
    
    if params.verbose:
        pbar2 = tqdm(range(params.simu_setting.ntest), total=params.simu_setting.ntest)
    else:
        pbar2 = range(params.simu_setting.ntest)
    with Parallel(n_jobs=params.n_jobs) as parallel:
        test_res = parallel(delayed(_run_fn2)(te_idx) for te_idx in pbar2)
    prbs = np.mean([res['in_sets'] for res in test_res], axis=0)
    mlen = np.mean([res['intvs_len'] for res in test_res])
        
        
    # results from CQR
    CQR_CIs = get_CQR_CIs(X=data_train.X, Y=data_train.Y, 
                          T=data_train.T, Xtest=data_test.X, 
                          nav=0, 
                          alpha=params.hypo_test.alpha, 
                          estimand="nonmissing",
                          fyx_est="quantBoosting", seed=manualSeed)
    mlen_cqr = np.mean(CQR_CIs[:, 1] -  CQR_CIs[:, 0])
    prb_Y1_cqr = np.bitwise_and(data_test.Y1>CQR_CIs[:, 0], data_test.Y1<CQR_CIs[:, 1]).mean()
    prb_tau_cqr = np.bitwise_and(data_test.tau>CQR_CIs[:, 0], data_test.tau<CQR_CIs[:, 1]).mean()
    
    res_all = edict()
    res_all.DDPM = (prbs, mlen)
    res_all.CQR = ([prb_Y1_cqr, prb_tau_cqr], mlen_cqr)
    return res_all


# In[ ]:


with Parallel(n_jobs=1) as parallel:
    test_ress = parallel(delayed(_run_fn)(rep_ix) for rep_ix in tqdm(range(params.nrep), total=params.nrep))

save_pkl((RES_ROOT/params.save_dir)/f"res_demo.pkl", test_ress, is_force=True)


