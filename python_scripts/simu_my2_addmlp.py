#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


from constants import RES_ROOT 
from utils.misc import  save_pkl 
from data_gen_utils.data_gen_my2 import get_simu_data
from utils.utils import MyDataSet, get_idx_sets
from demo_settings import simu_settings
from CQR import get_CQR_CIs, boosting_pred, boosting_logi, get_CF_CIs
from mlp.train_mlp import TrainMLP
from weighted_conformal_inference import WeightedConformalInference

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
parser.add_argument('--d', type=int, default=0, help='num of features') 
parser.add_argument('--n', type=int, default=0, help='sample size') 
parser.add_argument('--epoch', type=int, default=3000, help='epoch number') 
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate') 
args = parser.parse_args()


# # Params
setting = args.setting
params = edict()

params.simu_setting = simu_settings[setting]
params.simu_setting.cal_ratio = 0.25 # for conformal inference
params.simu_setting.val_ratio = 0.15 # for tuning network
if args.d > 0:
    params.simu_setting.d = args.d
if args.n > 0:
    params.simu_setting.n = args.n
params.simu_setting.ntest = 1000
pprint(params.simu_setting)


params.nrep = 50 # num of reptition for simulation
params.save_snapshot = 500
params.df_dtype = torch.float32
params.device="cpu"
params.n_jobs = 1
params.inf_bs = 40
params.verbose = False

params.mlp_training = edict()
# Batch size during training
params.mlp_training.batch_size = 256 
# Number of training epochs
params.mlp_training.n_epoch = args.epoch
params.mlp_training.n_infeat = 128
# Learning rate for optimizers
params.mlp_training.lr = 0.001
params.mlp_training.lr_gamma = 0.7
params.mlp_training.lr_step = 1000
params.mlp_training.test_intv = 500
params.mlp_training.n_downblk = 1
params.mlp_training.weight_decay = 1e-2
params.mlp_training.early_stop = False
params.mlp_training.early_stop_dict = {}

params.wconformal = edict()
# remove too large and too small in ws/mean(ws)
params.wconformal.nwthigh = 20
params.wconformal.nwtlow = 0.05
params.wconformal.useinf = False


params.hypo_test = edict()
params.hypo_test.alpha = 0.05 # sig level

params.prefix = ""
params.save_dir = f"simu_my2{setting}_d{params.simu_setting.d}_n{params.simu_setting.n}"
if not (RES_ROOT/params.save_dir).exists():
    (RES_ROOT/params.save_dir).mkdir()


# In[85]:


torch.set_default_dtype(params.df_dtype)




def _get_name_postfix(keys, mlp_training):
    lst = []
    for key in keys:
        if mlp_training[key] >= 1:
            lst.append(f"{key}-{str(mlp_training[key])}")
        else:
            lst.append(f"{key}--{str(mlp_training[key]).split('.')[-1]}")
    return "_".join(lst)
pprint(params)


def _main_fn(rep_ix, params, lr, n_infeat, weight_decay, n_blk):
    manualSeed = rep_ix
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    params = edict(deepcopy(params))
    params.mlp_training.n_downblk = n_blk
    params.mlp_training.weight_decay = weight_decay
    params.mlp_training.lr = lr
    params.mlp_training.n_infeat = n_infeat
    keys = ["lr", "n_infeat", "weight_decay", "n_downblk"]
    post_fix = _get_name_postfix(keys, params.mlp_training)
    
    torch.set_default_dtype(params.df_dtype)
    torch.set_default_device(params.device)
    
    fil_name = (RES_ROOT/params.save_dir)/f"rep_{rep_ix}_{post_fix}_mlp_res.pkl"
        
        
    data_train = get_simu_data(n=params.simu_setting.n, 
                                   d=params.simu_setting.d, 
                                   is_homo=params.simu_setting.is_homo, 
                                   rho=params.simu_setting.rho, 
                                   err_type=params.simu_setting.err_type);
    data_test = get_simu_data(n=params.simu_setting.ntest, 
                                   d=params.simu_setting.d, 
                                   is_homo=params.simu_setting.is_homo, 
                                   rho=params.simu_setting.rho,
                                   err_type=params.simu_setting.err_type);
    if not fil_name.exists():
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
            
        # train q(Y(1)|X)
        data_train = MyDataSet(Y=data_train.Y[tr_idxs], X=data_train.X[tr_idxs])
        data_val = edict()
        data_val.c = val_X
        data_val.x = val_Y
        input_params = edict(deepcopy(params.mlp_training))
        input_params.pop("n_epoch")
        input_params.pop("early_stop")
        input_params.pop("early_stop_dict")
        mymlp = TrainMLP(data_train, save_dir=params.save_dir, verbose=params.verbose, prefix=f"rep{rep_ix}_{post_fix}", 
                           device=params.device,
                           **input_params);
        
        
        mymlp.train(n_epoch=params.mlp_training.n_epoch, 
                    data_val=data_val, save_snapshot=params.save_snapshot, 
                    early_stop=params.mlp_training.early_stop, 
                    early_stop_dict=params.mlp_training.early_stop_dict
                         )
        def _inner_fn(X, Y, net, gen_type="reg"):
            wcf = WeightedConformalInference(cal_X, 
                                             cal_Y,
                                             net, ws_fn=wsfun, verbose=1, 
                                             gen_type=gen_type,
                                             seed=manualSeed,
                                             n_jobs=params.n_jobs,
                                             inf_bs=params.inf_bs,
                                             device=params.device,
                                             gen_params={},
                                             wcf_params={
                                                "nwhigh" : params.wconformal.nwthigh,
                                                "nwlow" : params.wconformal.nwtlow,
                                                "useinf": params.wconformal.useinf,
                                             })
            # get the len of CI based on intvs, there intvs is a list, each ele is another list contains CIs ele=[CI1, CI2]
            _get_intvs_len = lambda intvs: np.array([sum([np.diff(iv) for iv in intv])[0] for intv in intvs]);
            # get weather vaule in vs is in CI in intvs or not 
            def _get_inset(vs, intvs):
                in_set = []
                for v, intv in zip(vs, intvs):
                    in_set.append(np.sum([np.bitwise_and(v>iv[0], v<iv[1]) for iv in intv]))
                in_set = np.array(in_set)
                return in_set
            intvs = wcf(X, alpha=params.hypo_test.alpha)
            prbs = np.mean(_get_inset(Y, intvs))
            mlen = np.median(_get_intvs_len(intvs))
            return prbs, mlen
        
            
            
        
        # get the results
        res_all = edict()
        
        # results under the final model
        net = mymlp.nn_model
        net.eval()
        res_all.mlp = _inner_fn(test_X, test_Y, net, gen_type="reg")
        res_all.mlp_val = _inner_fn(val_X, val_Y, net, gen_type="reg")
           
        for model_ix in range(params.save_snapshot, params.mlp_training.n_epoch, params.save_snapshot):
            net = mymlp.get_model(model_ix)
            net.eval()
            res_all[f"mlp_ep{model_ix}"] = _inner_fn(test_X, test_Y, net, gen_type="reg")
            res_all[f"mlp_ep{model_ix}_val"] = _inner_fn(val_X, val_Y, net, gen_type="reg")
            
        save_pkl((RES_ROOT/params.save_dir)/fil_name, res_all, is_force=True)
        all_models = list(mymlp.save_dir.glob(f"{mymlp.prefix}mlp_epoch*.pth"))
        [m.unlink() for m in all_models]
    else:
        print(f"As {fil_name} exists, we do not do anything")
            
    return None



#lr, n_infeat, weight_decay, n_blk
lrs = [args.lr]
#lrs = [1e-1, 1e-2]
n_infeats = [128, 256]
n_blks = [1, 3]
weight_decays = [1e-2]
from itertools import product
all_coms = product(n_infeats, n_blks, lrs, weight_decays, range(params.nrep))
n_coms = params.nrep * len(lrs)  * len(n_infeats) * len(n_blks) * len(weight_decays)


with Parallel(n_jobs=35) as parallel:
    test_ress = parallel(delayed(_main_fn)(rep_ix, params=params, 
                                                  lr=lr, n_infeat=n_infeat, 
                                                  weight_decay=weight_decay, n_blk=n_blk) 
                         for n_infeat, n_blk, lr, weight_decay, rep_ix 
                         in tqdm(all_coms, total=n_coms))


