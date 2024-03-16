import numpy as np
from easydict import EasyDict as edict
def get_data_from_rawreal(data_raw, rep_ix, nrep):
    """Get the data for simulation based on IHDP data
    args:
        data_raw: the raw data loaded with np.load
        rep_ix (int): simulation index, from 0
        nrep (int): Num of simulations you want to do. As we have 1000 reps, so nrep <= 1000
    return:
        dataset: a dataset
    """
    ndata_tt = data_raw["x"].shape[-1]
    assert ndata_tt % nrep == 0
    ndata_rep = int(ndata_tt/nrep);
    lix, uix = rep_ix * ndata_rep,  (rep_ix+1) * ndata_rep
    X = data_raw["x"][:, :, lix:uix];
    X = X.transpose(2, 0, 1).reshape(-1, X.shape[1]); # num_sub x num_cov, order rep1, rep2,...
    
    Y = data_raw["yf"][:, lix:uix];
    Y = Y.T.flatten();
    
    T = data_raw["t"][:, lix:uix];
    T = T.T.flatten();
    
    Yunobs = data_raw["ycf"][:, lix:uix];
    Yunobs = Yunobs.T.flatten();
    
    Y1 = Y * (T==1) + Yunobs*(T==0);
    Y0 = Y * (T==0) + Yunobs*(T==1);
    
    tau = (data_raw["mu1"] - data_raw["mu0"])[:, lix:uix]
    tau = tau.T.flatten();
    
    dataset = edict()
    dataset.X = X
    dataset.Y = Y
    dataset.Y0 = Y0
    dataset.Y1 = Y1
    dataset.T = T
    dataset.tau = tau
    return dataset