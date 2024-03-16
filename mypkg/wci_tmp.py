import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from joblib import Parallel, delayed
from utils.utils import _set_verbose_level, _update_params
from utils.misc import merge_intervals
from utils.stats import weighted_quantile
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

class ConformalInference():
    def __init__(self, cal_X, cal_Y, gen_fn, 
                 verbose=2,device="cpu", wcf_params={}, 
                 gen_params={}):
        """
        args:
            cal_X (tensor, n_cal x d):  The covariates of the calibration set. 
            cal_Y (tensor, n_cal):  The response of the calibration set. 
            gen_fn (fn): A generating function f(Y|X), given X, it can produce predicted Y.
        """
        _set_verbose_level(verbose, logger)
        wcf_params_def = edict({
            "nwhigh" : 20,
            "nwlow" : 0.05,
            "useinf" :False, 
        })
        gen_parmas_def = edict({
        })
        wcf_params_def["cf_type"] = "naive"
        
        
        wcf_params = _update_params(wcf_params, wcf_params_def, logger)
        gen_params = _update_params(gen_params, gen_parmas_def, logger)
        logger.info(f"wcf params is {wcf_params}")
        logger.info(f"gen params is {gen_params}")
        
        
 
        def _gen_fn(X):
            Y = gen_fn(X)
            return Y.to(device).squeeze()
            
        
        self._gen_fn = _gen_fn
        self.verbose = verbose
        self.wcf_params = wcf_params
        self.gen_params = gen_params
        self.df_dtype = torch.get_default_dtype()
        self.device = device
        if not isinstance(cal_X, torch.Tensor):
            cal_X = torch.tensor(cal_X, dtype=self.df_dtype)
            cal_Y = torch.tensor(cal_Y, dtype=self.df_dtype)
        self.cal_X = cal_X
        self.cal_Y = cal_Y
        
        self.Es = None

        
    
    def naive_dist_fn(self):
        cal_Y_preds = self._gen_fn(self.cal_X)
        Es = torch.abs(cal_Y_preds.squeeze()-self.cal_Y);
        return Es
    
    def naive_intv_fn(self, Y_hat, qvs):
        intvs_raw = torch.stack([Y_hat-qvs, Y_hat+qvs]).T
        intvs_raw = list(intvs_raw.numpy())
        return intvs_raw
    def get_intvs(self, Y_hat, qvs):
        return self.naive_intv_fn(Y_hat, qvs)
    
    def get_dist(self):
        self.Es = self.naive_dist_fn()
        return self.Es
    
    
