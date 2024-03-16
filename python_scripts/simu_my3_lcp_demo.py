#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


from constants import RES_ROOT 
from utils.misc import  save_pkl 
from data_gen_utils.data_gen_my3 import get_simu_data
from utils.utils import MyDataSet, get_idx_sets
from demo_settings import simu_settings
from CQR import get_CQR_CIs, boosting_pred, boosting_logi, get_CF_CIs
from ddpm.train_ddpm_now import TrainDDPM
from weighted_conformal_inference import WeightedConformalInference
from local_weighted_conformal_inference import LocalWeightedConformalInference, get_opth
from mlp.train_mlp import TrainMLP
from naive_sample import NaiveSample

import torch
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm, trange
import random
from joblib import Parallel, delayed
from pprint import pprint
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser(description='run simu under causal context for lei setting')
parser.add_argument('-s', '--setting', type=str, default="setting1", help='the simu setting') 
parser.add_argument('--h', type=float, default=0.5, help='bw of kernel') 
args = parser.parse_args()

# # Params
setting = args.setting
#setting = "setting3"
params = edict()

params.simu_setting = simu_settings[setting]
params.simu_setting.cal_ratio = 0.25 # for conformal inference
params.simu_setting.val_ratio = 0.15 # for tuning network
params.simu_setting.d = 10
params.simu_setting.n = 3000
params.simu_setting.ntest = 1000
pprint(params.simu_setting)


params.nrep = 50 # num of reptition for simulation
params.K = 40 # num of sps drawn from q(Y(1)|X)
params.save_snapshot = False
params.df_dtype = torch.float32
params.device="cpu"
params.n_jobs = 1
params.verbose = False
params.inf_bs = 250 # the inference batch in number of X, so the real bs is inf_bs x K

params.ddpm_training = edict()
# Batch size during training
params.ddpm_training.batch_size = 256 
# Number of training epochs
params.ddpm_training.n_epoch = 2000
params.ddpm_training.n_infeat = 128
# Learning rate for optimizers
params.ddpm_training.lr = 0.001
params.ddpm_training.lr_gamma = 0.7
params.ddpm_training.lr_step = 1000
params.ddpm_training.test_intv = 500
params.ddpm_training.n_T = 400 # 100
params.ddpm_training.n_upblk = 1
params.ddpm_training.n_downblk = 1
params.ddpm_training.weight_decay = 1e-2
params.ddpm_training.early_stop = False
params.ddpm_training.early_stop_dict = {}
#params.ddpm_training.betas = [0.001, 0.5]

params.wconformal = edict()
# remove too large and too small in ws/mean(ws)
params.wconformal.nwthigh = 20
params.wconformal.nwtlow = 0.05
params.wconformal.useinf = False


params.hypo_test = edict()
params.hypo_test.alpha = 0.05 # sig level

params.prefix = ""
params.save_dir = f"simuLCP_my3{setting}_h{args.h*100:.0f}"
if not (RES_ROOT/params.save_dir).exists():
    (RES_ROOT/params.save_dir).mkdir()


# In[85]:


torch.set_default_dtype(params.df_dtype)




#keys = ["lr", "n_infeat", "n_T", "weight_decay", "n_upblk", "n_downblk"]
def _get_name_postfix(keys, ddpm_training):
    lst = []
    for key in keys:
        if ddpm_training[key] >= 1:
            lst.append(f"{key}-{str(ddpm_training[key])}")
        else:
            lst.append(f"{key}--{str(ddpm_training[key]).split('.')[-1]}")
    return "_".join(lst)
pprint(params)


def _main_fn(rep_ix, params, lr, n_infeat, n_T, weight_decay, n_blk):
    manualSeed = rep_ix
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    params = edict(deepcopy(params))
    params.ddpm_training.n_upblk = n_blk
    params.ddpm_training.n_downblk = n_blk
    params.ddpm_training.weight_decay = weight_decay
    params.ddpm_training.n_T = n_T
    params.ddpm_training.lr = lr
    params.ddpm_training.n_infeat = n_infeat
    keys = ["lr", "n_infeat", "n_T", "weight_decay", "n_upblk", "n_downblk"]
    post_fix = _get_name_postfix(keys, params.ddpm_training)
    
    torch.set_default_dtype(params.df_dtype)
    torch.set_default_device(params.device)
    
    fil_name = (RES_ROOT/params.save_dir)/f"rep_{rep_ix}_{post_fix}_res.pkl"
    ofil_name = (RES_ROOT/params.save_dir)/f"rep_{rep_ix}_other_res.pkl"
        
        
    data_train = get_simu_data(n=params.simu_setting.n, 
                                   d=params.simu_setting.d, 
                                   rho=params.simu_setting.rho, 
                                   err_type=params.simu_setting.err_type);
    data_test = get_simu_data(n=params.simu_setting.ntest, 
                                   d=params.simu_setting.d, 
                                   rho=params.simu_setting.rho,
                                   err_type=params.simu_setting.err_type);
    cal_idxs, val_idxs, tr_idxs = get_idx_sets(all_idxs=np.where(data_train.T)[0], 
                                                   ratios = [params.simu_setting.cal_ratio, params.simu_setting.val_ratio])
            
    # get wsfun, note that ws is 1/ps
    psY = data_train.T.astype(int)
    psX = data_train.X
    fit_res = boosting_logi(psY, psX);
    def wsfun(X):
        eps = 1e-10
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        est_ws = 1/(boosting_pred(X, fit_res) + eps)
        return torch.tensor(est_ws, dtype=params.df_dtype).to(device=params.device)
        
    
    cal_X = torch.tensor(data_train.X[cal_idxs], dtype=params.df_dtype)
    cal_Y = torch.tensor(data_train.Y1[cal_idxs], dtype=params.df_dtype)
    val_X = torch.tensor(data_train.X[val_idxs], dtype=params.df_dtype)
    val_Y = torch.tensor(data_train.Y1[val_idxs], dtype=params.df_dtype)
    test_X = torch.tensor(data_test.X, dtype=params.df_dtype)
    test_Y = torch.tensor(data_test.Y1, dtype=params.df_dtype)
        
    # get subset of all X
    test_Xnorm = torch.norm(test_X, p=2, dim=1)
    cutoff1 = torch.quantile(test_Xnorm, 0.25)
    cutoff2 = torch.quantile(test_Xnorm, 0.5)
    test_X1 = test_X[test_Xnorm<cutoff1].clone()
    test_Y1 = test_Y[test_Xnorm<cutoff1].clone()
    test_X2 = test_X[test_Xnorm<cutoff2].clone()
    test_Y2 = test_Y[test_Xnorm<cutoff2].clone()
    test_X2c = test_X[test_Xnorm>=cutoff2].clone()
    test_Y2c = test_Y[test_Xnorm>=cutoff2].clone();

    if not fil_name.exists():
            
        # train q(Y(1)|X)
        data_train_ddpm = MyDataSet(Y=data_train.Y[tr_idxs], X=data_train.X[tr_idxs])
        data_val = edict()
        data_val.c = val_X
        data_val.x = val_Y
        input_params = edict(deepcopy(params.ddpm_training))
        input_params.pop("n_epoch")
        input_params.pop("early_stop")
        input_params.pop("early_stop_dict")
        myddpm = TrainDDPM(data_train_ddpm, save_dir=params.save_dir, verbose=params.verbose, prefix=f"rep{rep_ix}_{post_fix}", 
                           device=params.device,
                           **input_params);
        
        
        myddpm.train(n_epoch=params.ddpm_training.n_epoch, 
                         data_val=data_val, save_snapshot=params.save_snapshot, 
                         early_stop=params.ddpm_training.early_stop, 
                         early_stop_dict=params.ddpm_training.early_stop_dict
                         )
        # train q(Y(1)|X) on MLP
        data_train_mlp = MyDataSet(Y=data_train.Y[tr_idxs], X=data_train.X[tr_idxs])
        data_val = edict()
        data_val.c = val_X
        data_val.x = val_Y
        input_params = edict(deepcopy(params.ddpm_training))
        input_params.pop("n_epoch")
        input_params.pop("early_stop")
        input_params.pop("early_stop_dict")
        input_params.pop("n_T")
        input_params.pop("n_upblk")
        mymlp = TrainMLP(data_train_mlp, save_dir=params.save_dir, verbose=params.verbose, prefix=f"rep{rep_ix}_{post_fix}", 
                           device=params.device,
                           **input_params);
        
        
        mymlp.train(n_epoch=params.ddpm_training.n_epoch, 
                    data_val=data_val, save_snapshot=params.save_snapshot, 
                    early_stop=params.ddpm_training.early_stop, 
                    early_stop_dict=params.ddpm_training.early_stop_dict
                         )

        def _inner_fn(X, Y, ddpm, gen_type="ddpm", gen_params={"ddim_eta": 1}, LCP=False):
            # get the len of CI based on intvs, there intvs is a list, each ele is another list contains CIs ele=[CI1, CI2]
            _get_intvs_len = lambda intvs: np.array([sum([np.diff(iv) for iv in intv])[0] for intv in intvs]);
            # get weather vaule in vs is in CI in intvs or not 
            def _get_inset(vs, intvs):
                in_set = []
                for v, intv in zip(vs, intvs):
                    in_set.append(np.sum([np.bitwise_and(v>iv[0], v<iv[1]) for iv in intv]))
                in_set = np.array(in_set)
                return in_set
            wcf = LocalWeightedConformalInference(cal_X, 
                                     cal_Y,
                                     ddpm, ws_fn=wsfun, verbose=1, 
                                     gen_type=gen_type,
                                     seed=manualSeed,
                                     n_jobs=params.n_jobs,
                                     inf_bs=params.inf_bs,
                                     device=params.device,
                                     gen_params=gen_params,
                                     wcf_params={
                                        "K": params.K, # num of sps for each X
                                        "nwhigh" : params.wconformal.nwthigh,
                                        "nwlow" : params.wconformal.nwtlow,
                                        "useinf": params.wconformal.useinf,
                                     })
            wcf.add_data(X)
            if not LCP:
                intvs = wcf(local_method=None, alpha=params.hypo_test.alpha);
            else:
                intvs = wcf(local_method="g-RLCP", alpha=params.hypo_test.alpha, lm_params={"h":args.h});
            prbs = np.mean(_get_inset(Y, intvs))
            mlen = np.median(_get_intvs_len(intvs))
            return prbs, mlen
        
        # get the results
        res_all = edict()
        
        # results under the final model
        ddpm = myddpm.ddpm
        ddpm.eval()
        res_all.DDIM = _inner_fn(test_X, test_Y, ddpm, gen_type="ddim", LCP=False)
        res_all.DDIM1 = _inner_fn(test_X1, test_Y1, ddpm, gen_type="ddim", LCP=False)
        res_all.DDIM2 = _inner_fn(test_X2, test_Y2, ddpm, gen_type="ddim", LCP=False)
        res_all.DDIM2c = _inner_fn(test_X2c, test_Y2c, ddpm, gen_type="ddim", LCP=False)
        
        res_all.LDDIM = _inner_fn(test_X, test_Y, ddpm, gen_type="ddim", LCP=True)
        res_all.LDDIM1 = _inner_fn(test_X1, test_Y1, ddpm, gen_type="ddim", LCP=True)
        res_all.LDDIM2 = _inner_fn(test_X2, test_Y2, ddpm, gen_type="ddim", LCP=True)
        res_all.LDDIM2c = _inner_fn(test_X2c, test_Y2c, ddpm, gen_type="ddim", LCP=True)

        net = mymlp.nn_model
        net.eval()
        res_all.MLP = _inner_fn(test_X, test_Y, net, gen_type="reg", LCP=False)
        res_all.MLP1 = _inner_fn(test_X1, test_Y1, net, gen_type="reg", LCP=False)
        res_all.MLP2 = _inner_fn(test_X2, test_Y2, net, gen_type="reg", LCP=False)
        res_all.MLP2c = _inner_fn(test_X2c, test_Y2c, net, gen_type="reg", LCP=False)
        
        res_all.LMLP = _inner_fn(test_X, test_Y, net, gen_type="reg", LCP=True)
        res_all.LMLP1 = _inner_fn(test_X1, test_Y1, net, gen_type="reg", LCP=True)
        res_all.LMLP2 = _inner_fn(test_X2, test_Y2, net, gen_type="reg", LCP=True)
        res_all.LMLP2c = _inner_fn(test_X2c, test_Y2c, net, gen_type="reg", LCP=True)
           
            
            
        save_pkl((RES_ROOT/params.save_dir)/fil_name, res_all, is_force=True)
    else:
        print(f"As {fil_name} exists, we do not do anything")
            
    if not ofil_name.exists():
        ## results from other methods
        res_other = edict()
        # results from CQR
        def _CQR_fn(test_X, test_Y):
            if isinstance(test_X, torch.Tensor):
                test_X = test_X.cpu().numpy()
            if isinstance(test_Y, torch.Tensor):
                test_Y = test_Y.cpu().numpy()
            CQR_CIs = get_CQR_CIs(X=data_train.X, Y=data_train.Y, 
                                  T=data_train.T, Xtest=test_X, 
                                  nav=0, 
                                  alpha=params.hypo_test.alpha, 
                                  estimand="unconditional",
                                  fyx_est="quantBoosting", seed=manualSeed)
            mlen_cqr = np.median(CQR_CIs[:, 1] -  CQR_CIs[:, 0])
            prb_Y1_cqr = np.bitwise_and(test_Y>CQR_CIs[:, 0],test_Y<CQR_CIs[:, 1]).mean()
            return prb_Y1_cqr, mlen_cqr
        res_other.CQR = _CQR_fn(test_X, test_Y)
        res_other.CQR1 = _CQR_fn(test_X1, test_Y1)
        res_other.CQR2 = _CQR_fn(test_X2, test_Y2)
        res_other.CQR2c = _CQR_fn(test_X2c, test_Y2c)
        

        save_pkl((RES_ROOT/params.save_dir)/ofil_name, res_other, is_force=True)
    else:
        print(f"As {ofil_name} exists, we do not do anything")
    return None



#lr, n_infeat, n_T, weight_decay, n_blk
# based on results, remove lr=0.5
lrs = [1e-2]
n_Ts = [400]
n_infeats = [128]
n_blks = [1]
weight_decays = [1e-2]
from itertools import product
all_coms = product(n_Ts, n_infeats, n_blks, lrs, weight_decays, range(params.nrep))
n_coms = params.nrep * len(lrs) * len(n_Ts) * len(n_infeats) * len(n_blks) * len(weight_decays)


with Parallel(n_jobs=35) as parallel:
    test_ress = parallel(delayed(_main_fn)(rep_ix, params=params, 
                                                  lr=lr, n_T=n_T, n_infeat=n_infeat, 
                                                  weight_decay=weight_decay, n_blk=n_blk) 
                         for n_T, n_infeat, n_blk, lr, weight_decay, rep_ix 
                         in tqdm(all_coms, total=n_coms))


