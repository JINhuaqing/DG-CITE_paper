from constants import RES_ROOT
from models.ddpm import ContextNet, ddpm_schedules, DDPM
import torch
import numpy as np
from easydict import EasyDict as edict
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR 
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
class TrainDDPM():
    def __init__(self, data_train, save_dir=None, prefix="", verbose=False, **input_params):
        super().__init__()
        if verbose:
            logger.handlers[0].setLevel(logging.INFO)
        else:
            logger.handlers[0].setLevel(logging.WARNING)
        
        if save_dir is not None:
            save_dir = RES_ROOT/save_dir
            if not save_dir.exists():
                save_dir.mkdir()
            logger.info(f"The results are saved at {save_dir}.")
        if len(prefix) > 0 and (not prefix.endswith("_")):
            prefix = prefix + "_"
        
        params_ddpm = edict({
            "lr": 1e-4, 
            "batch_size": 128, 
            "device": "cpu",
            "n_T": 400, 
            "n_infeat": 256,
            "n_upblk": 1, 
            "n_downblk": 1, 
            "betas": (1e-4, 0.02),
            "drop_prob": 0.1, 
            # lr decreasing step factor
            "lr_gamma": 0.9,
            # lr decreasing step time, in epoch
            "lr_step": 1000, 
            # in epoch
            "test_intv": 100, 
        })
        
        for key in input_params.keys():
            if key not in params_ddpm.keys():
                logger.warning(f"{key} is not used, please check your input.")
            else:
                params_ddpm[key] = input_params[key]
        
        # get the MLP
        nn_model=ContextNet(in_fs=1, 
                                 n_infeat=params_ddpm.n_infeat, 
                                 n_context_fs=data_train[0][0].shape[0], 
                                 n_upblk=params_ddpm.n_upblk, 
                                 n_downblk=params_ddpm.n_downblk
                                )
        nn_model.weights_init()
        if verbose:
            nn_model.get_num_params()
        
        # get the main class for training and inference
        self.ddpm = DDPM(nn_model=nn_model, 
                         betas=params_ddpm.betas, 
                         n_T=params_ddpm.n_T, 
                         device=params_ddpm.device, 
                         drop_prob=params_ddpm.drop_prob, 
                         verbose=verbose);
        self.ddpm.to(params_ddpm.device);
        
        # the optim and loss for training
        self.optim = torch.optim.Adam(self.ddpm.parameters(), lr=params_ddpm.lr);
        self.lr_scheduler = ExponentialLR(self.optim, gamma=params_ddpm.lr_gamma, verbose=verbose) 
        
        if verbose:
            logger.info(f"The params is {params_ddpm}")
            
        self.data_train_loader = DataLoader(data_train, batch_size=params_ddpm.batch_size, shuffle=True)
        self.params_ddpm = params_ddpm
        self.dftype = torch.get_default_dtype();
        self.verbose = verbose
        self.losses = [] # loss
        self.losses_sm = [] # smoother version of loss
        self.losses_test = [] # test error
        self.prefix = prefix
        self.save_dir = save_dir
            
            
    def train(self, n_epoch, data_test=None, save_snapshot=True):
        """args
            n_epoch: num of epochs
            data_test: a edict including
                c: bs x d array, feature
                x: bs/ bsx1, target 
        """
        assert not (save_snapshot and (self.save_dir is None)), "if you want to save model, plz provide save dir"
        if self.verbose:
            pbar = trange(n_epoch)
        else:
            pbar = range(n_epoch)
        
        if data_test is not None:
            c_test = torch.tensor(data_test.c, dtype=self.dftype).to(self.params_ddpm.device)
            x_test = torch.tensor(data_test.x, dtype=self.dftype).to(self.params_ddpm.device)
            
        for ep in pbar:
            self.ddpm.train()
            loss_sm = None # smoother version of loss
            for data in self.data_train_loader:
                self.optim.zero_grad()
                c, x = data # c is features, x is target
                if x.ndim == 1:
                    x = x[:, None]
                c, x = c.to(self.params_ddpm.device), x.to(self.params_ddpm.device)
                c, x = c.type(self.dftype), x.type(self.dftype)
                b_size = x.shape[0] # note it can be different from the batch_size
                if b_size == 1:
                    # let us ignore the batch with size 1, cause error (on Dec 8, 2023)
                    continue
                    
                loss = self.ddpm(x, c)
                loss.backward()
                if loss_sm is None:
                    loss_sm = loss.item()
                else:
                    loss_sm = 0.95 * loss_sm + 0.05 * loss.item()
                if self.verbose:
                    pbar.set_description(f"loss: {loss_sm:.4f}")
                self.optim.step()
                self.losses.append(loss.item())
                self.losses_sm.append(loss_sm)
                
            if data_test is not None:
                if (ep+1)%self.params_ddpm.test_intv == 0:
                    self.ddpm.eval()
                    with torch.no_grad():
                        loss_test = self.ddpm(x_test, c_test)
                    self.ddpm.train()
                    self.losses_test.append((ep+1, loss_test.item()))
                    if self.verbose:
                        out_dict = {"test loss": loss_test.item()}
                        pbar.set_postfix(out_dict, refresh=True)
                
            
            if (ep+1) % self.params_ddpm.lr_step == 0:
                # change lr
                self.lr_scheduler.step()
                
    
            if save_snapshot:
                if isinstance(save_snapshot, bool):
                    save_int = n_epoch 
                else:
                    save_int = save_snapshot
                if (ep+1) % save_int == 0:
                    logger.info(f"Save model {self.prefix}ddpm_epoch{ep+1}.pth.")
                    torch.save(self.ddpm.state_dict(), self.save_dir/f"{self.prefix}ddpm_epoch{ep+1}.pth")