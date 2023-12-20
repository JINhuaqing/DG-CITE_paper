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
parser.add_argument('-s', '--setting', type=str, default="setting3", help='the simu setting') 
parser.add_argument('--d', type=int, default=0, help='num of features') 
parser.add_argument('--n', type=int, default=0, help='sample size') 
parser.add_argument('--epoch', type=int, default=2000, help='epoch number') 
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
if args.d > 0:
    params.simu_setting.d = args.d
if args.n > 0:
    params.simu_setting.n = args.n
pprint(params.simu_setting)


params.nrep = 50 # num of reptition for simulation
params.K = 40 # num of sps drawn from q(Y(1)|X)
params.save_snapshot = 100
params.dftype = torch.float32
params.device="cpu"
params.n_jobs = 1
params.verbose = False
params.inf_bs = 50 # the inference batch, fct x K

params.ddpm_training = edict()
# Batch size during training
params.ddpm_training.batch_size = 256 
# Number of training epochs
params.ddpm_training.n_epoch = args.epoch
params.ddpm_training.n_infeat = 128
# Learning rate for optimizers
params.ddpm_training.lr = 0.001
params.ddpm_training.lr_gamma = 0.5
params.ddpm_training.lr_step = 1000
params.ddpm_training.test_intv = 5
params.ddpm_training.n_T = 200 # 100
params.ddpm_training.drop_prob = 0.0
params.ddpm_training.n_upblk = 1
params.ddpm_training.n_downblk = 1
params.ddpm_training.weight_decay = 1e-2
#params.ddpm_training.betas = [0.001, 0.5]

params.wconformal = edict()
# remove too large and too small in ws/mean(ws)
params.wconformal.nwthigh = 20
params.wconformal.nwtlow = 0.05
params.wconformal.useinf = False


params.hypo_test = edict()
params.hypo_test.alpha = 0.05 # sig level

params.prefix = ""
params.save_dir = f"demodp0_ablation_{setting}_d{params.simu_setting.d}_n{params.simu_setting.n}"
if not (RES_ROOT/params.save_dir).exists():
    (RES_ROOT/params.save_dir).mkdir()    



keys = ["lr", "n_infeat", "n_T", "weight_decay", "n_upblk", "n_downblk"]
def _get_name_postfix(keys):
    return "_".join([f"{key}-{str(params.ddpm_training[key]).split('.')[-1]}" for key in keys])


pprint(params)



# # Some fns

# In[14]:


def _gen_Y_given_X(X, ddpm, seed=1):
    c_all = torch.tensor(X, dtype=params.dftype).to(params.device);
    num_iters = int(np.ceil(c_all.shape[0]/params.inf_bs));
    x_0s = []
    if params.verbose:
        pbar = tqdm(range(num_iters), total=num_iters)
    else:
        pbar = range(num_iters)
    def _run_fn(ix):
        torch.set_default_dtype(params.dftype)
        torch.set_default_device(params.device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        c_cur = c_all[(ix*params.inf_bs):(ix*params.inf_bs+params.inf_bs)]
        c_cur_mul = c_cur.repeat(params.K, 1);
        with torch.no_grad():
            x_0, _ = ddpm.sample(c_cur_mul, device=params.device, guide_w=0, is_store=False);
        x_0 = x_0.cpu().numpy().reshape(-1);
        x_0 = x_0.reshape(params.K, -1);
        return x_0
    with Parallel(n_jobs=params.n_jobs) as parallel:
        x_0s = parallel(delayed(_run_fn)(ix) for ix in pbar)
    x_0s = np.concatenate(x_0s, axis=1).T;
    return x_0s
def _get_pred_intv(teY_hat, qv):
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

def _run_fn_quanreg(rep_ix, params, lr, n_infeat, n_T, weight_decay, n_blk):
    manualSeed = rep_ix
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    params = edict(params.copy())
    params.ddpm_training.n_upblk = n_blk
    params.ddpm_training.n_downblk = n_blk
    params.ddpm_training.weight_decay = weight_decay
    params.ddpm_training.n_T = n_T
    params.ddpm_training.lr = lr
    params.ddpm_training.n_infeat = n_infeat
    post_fix = _get_name_postfix(keys)
    
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
    myddpm = TrainDDPM(data_train_ddpm, save_dir=params.save_dir, verbose=params.verbose, prefix=f"rep{rep_ix}_{post_fix}", 
                       device=params.device,
                       **input_params);
    
    
    def _inner_fn(ddpm):
        # weight function
        def wtfun(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            # for unconditional weight
            return np.ones(x.shape[0])
            
        # get the weight and nonconformity score for each data point in cal set
        x_0s = _gen_Y_given_X(data_train.X[cal_idxs], ddpm, manualSeed)
        ws = wtfun(data_train.X[cal_idxs]);
        Es = np.abs(x_0s -  data_train.Y1[cal_idxs][:, None]).min(axis=1);
                
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
            qt = (1 + nws_wtest[-1]/ nws_wtest[:-1].sum()) * (1 - params.hypo_test.alpha)
            if qt <= 1:
                qv_cur = weighted_quantile(Es, [qt], sample_weight=ws_wtest[:-1])[0];
            else:
                if params.wconformal.useinf:
                    qv_cur = np.inf
                else:
                    qv_cur = np.max(Es)
            
            intvs = _get_pred_intv(teYs_hat[te_idx], qv_cur)
            res = _get_metric([teY1, tetau], intvs)
            res["qv_cur"] = qv_cur
            res["qt"] = qt
            res["intvs"] = intvs
            return res
            
        teYs_hat = _gen_Y_given_X(data_test.X, ddpm, manualSeed);
        pbar2 = range(params.simu_setting.ntest)
        with Parallel(n_jobs=1) as parallel:
            test_res = parallel(delayed(_run_fn2)(te_idx) for te_idx in pbar2)
            
        prbs = np.mean([res['in_sets'] for res in test_res], axis=0)
        mlen = np.median([res['intvs_len'] for res in test_res])
        return prbs, mlen
    
    myddpm.train(n_epoch=params.ddpm_training.n_epoch, 
                     data_val=data_val, save_snapshot=params.save_snapshot)
    ddpm = myddpm.ddpm
    ddpm.eval()
    prbs1, mlen1 = _inner_fn(ddpm)
    ddpm = myddpm.get_opt_model()
    ddpm.eval()
    prbs2, mlen2 = _inner_fn(ddpm)
        
        
    # results from CQR
    CQR_CIs = get_CQR_CIs(X=data_train.X, Y=data_train.Y, 
                          T=data_train.T, Xtest=data_test.X, 
                          nav=0, 
                          alpha=params.hypo_test.alpha, 
                          estimand="nonmissing",
                          fyx_est="quantBoosting", seed=manualSeed)
    mlen_cqr = np.median(CQR_CIs[:, 1] -  CQR_CIs[:, 0])
    prb_Y1_cqr = np.bitwise_and(data_test.Y1>CQR_CIs[:, 0], data_test.Y1<CQR_CIs[:, 1]).mean()
    prb_tau_cqr = np.bitwise_and(data_test.tau>CQR_CIs[:, 0], data_test.tau<CQR_CIs[:, 1]).mean()
    
    res_all = edict()
    res_all.DDPM = (prbs1, mlen1)
    res_all.DDPM_sel = (prbs2, mlen2)
    res_all.CQR = ([prb_Y1_cqr, prb_tau_cqr], mlen_cqr)
    save_pkl((RES_ROOT/params.save_dir)/f"rep_{rep_ix}_{post_fix}_res.pkl", res_all, is_force=True)
    all_models = list(myddpm.save_dir.glob(f"{myddpm.prefix}ddpm_epoch*.pth"))
    [m.unlink() for m in all_models]
    return None



# In[ ]:
#lr, n_infeat, n_T, weight_decay, n_blk
lrs = [1e-2, 1e-3, 1e-4]
n_Ts = [100, 400, 1000]
n_infeats = [128, 512]
n_blks = [1, 3]
weight_decays = [1e-2]
from itertools import product
all_coms = product(n_Ts, n_infeats, n_blks, lrs, weight_decays, range(params.nrep))


with Parallel(n_jobs=35) as parallel:
    test_ress = parallel(delayed(_run_fn_quanreg)(rep_ix, params=params, 
                                                  lr=lr, n_T=n_T, n_infeat=n_infeat, 
                                                  weight_decay=weight_decay, n_blk=n_blk) 
                         for n_T, n_infeat, n_blk, lr, weight_decay, rep_ix 
                         in tqdm(all_coms, total=params.nrep*3*3*2*2*1))


