from tqdm import trange, tqdm
from constants import RES_ROOT
import scipy.stats as ss
import numpy as np
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pdb
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

#torch.set_default_dtype(torch.float64)
#torch.set_default_tensor_type(torch.DoubleTensor)




# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class MyGAN():
    def __init__(self, Generator, Discriminator, d, train_data, save_dir, 
                 nz=10, 
                 prefix='', verbose=False, **input_params):
        self.verbose = verbose
        if self.verbose:
            logger.handlers[0].setLevel(logging.INFO)
        else:
            logger.handlers[0].setLevel(logging.WARNING)
        save_dir = RES_ROOT/save_dir
        if not save_dir.exists():
            save_dir.mkdir()
        logger.info(f"The results are saved at {save_dir}.")
        if len(prefix) > 0 and (not prefix.endswith("_")):
            prefix = prefix + "_"
        self.prefix = prefix
        self.save_dir = save_dir
        self.d = d
        # Size of z latent vector (i.e. size of generator input)
        self.nz = nz
        self.params_gan = edict({
            # Batch size during training
            "batch_size": 32, 
            # Learning rate for optimizers
            "lr": 0.002,
            # Beta1 hyperparameter for Adam optimizers
            "beta1": 0.5, 
            "device": "cpu"
        })
        
        for key in input_params.keys():
            if key not in self.params_gan.keys():
                logger.warning(f"{key} is not used, please check your input.")
            else:
                self.params_gan[key] = input_params[key]
                
        self.train_data_loader = DataLoader(train_data, batch_size=self.params_gan.batch_size, shuffle=True)
        self.netG = Generator(self.d, self.nz);
        self.netD = Discriminator(self.d);
        self.netG.apply(weights_init);
        self.netD.apply(weights_init);
        
        # Initialize the ``BCELoss`` function
        self.criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 0.9
        self.fake_label = 0.1
        
        self.G_losses = None
        self.D_losses = None
        self.test_errs = None
        self.Generator = Generator
        
        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.params_gan.lr, betas=(self.params_gan.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.params_gan.lr, betas=(self.params_gan.beta1, 0.999));
        logger.info(f"The params are {self.params_gan}")

                
    def train(self, num_epochs, data_test=None, save_snapshot=True):
        if self.verbose:
            pbar = trange(num_epochs)
        else:
            pbar = range(num_epochs)
        # Lists to keep track of progress
        G_losses = []
        D_losses = []
        test_errs = []
        iters = 0
        
        # For each epoch
        self.netG.train()
        self.netD.train()
        for epoch in pbar:
            # For each batch in the dataloader
            for i, data in enumerate(self.train_data_loader, 0):
                X, realY = data
                realY = realY.unsqueeze(-1)
                b_size = realY.shape[0] # note it can be different from the batch_size
                if b_size == 1:
                    # let us ignore the batch with size 1, cause error (on Dec 8, 2023)
                    continue
        
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                label = torch.full((b_size,), self.real_label, 
                                   dtype=torch.float64, 
                                   device=self.params_gan.device)
                # Forward pass real batch through D
                output = self.netD(X, realY).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
        
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, device=self.params_gan.device)
                # Generate fake image batch with G
                fake = self.netG(X, noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(X, fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()
        
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(X, fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                if self.verbose:
                    if (iters+1) % 50 == 0:
                        out_dict = {"Loss_D": errD.item(), 
                                    "Loss_G": errG.item(), 
                                    "D(x)": D_x, 
                                    "D(G(z))": f"{D_G_z1:.3f}/{D_G_z2:.3f}"
                                    }
                        
                        if data_test is not None:
                            tX = torch.tensor(data_test.X)
                            tZ = torch.randn(data_test.X.shape[0], self.nz);
                        
                            self.netG.eval()
                            with torch.no_grad():
                                tY_hat = self.netG(tX, tZ).detach().numpy().reshape(-1);
                                test_err = np.sqrt(np.mean((tY_hat - data_test.Y1)**2))
                                v, _ = ss.pearsonr(data_test.Y1, tY_hat)
                            self.netG.train()
                            out_dict["test_err"] = test_err
                            out_dict["test_r"] = v
                            test_errs.append(test_err)
                                
                        pbar.set_postfix(out_dict, refresh=True)
                        
            
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())


                iters += 1
            if save_snapshot:
                if isinstance(save_snapshot, bool):
                    save_int = num_epochs
                else:
                    save_int = save_snapshot
                if (epoch+1) % save_int == 0:
                    logger.info(f"Save model {self.prefix}gan_G/D_epoch{epoch+1}_iter{iters}.pth.")
                    torch.save(self.netG.state_dict(), self.save_dir/f"{self.prefix}gan_G_epoch{epoch+1}_iter{iters}.pth")
                    torch.save(self.netD.state_dict(), self.save_dir/f"{self.prefix}gan_D_epoch{epoch+1}_iter{iters}.pth")
            self.G_losses = np.array(G_losses)
            self.D_losses = np.array(D_losses)
            self.test_errs = np.array(test_errs)
            
        self.netG.eval()
        self.netD.eval()
        
    def get_opt_model(self, ws=None, m_tol=0.05, sd_tol=0.05):
        """Select the best model based on the clossness to the theoretical loss values
           D_loss = -2*log(0.5) = 1.386
           G_loss = -log(0.5) = 0.693
           args: 
               ws: window size
               m_tol: tolorance of the mean deviating from theoretical value
               sd_tol: tolorance of the sd. 
        """
        all_models = list(self.save_dir.glob(f"{self.prefix}gan_G_*.pth"))
        assert (self.D_losses is not None) and (self.G_losses is not None) and (len(all_models) > 0), "You should run train() and save model first"
        _tmpf = lambda mp: int(mp.stem.split("iter")[-1])
        all_models = sorted(all_models, key=_tmpf)
        niters = [_tmpf(m) for m in all_models]
        if ws is None:
            ws = int(np.diff(niters)[0]/3)
            logger.info(f"The window size is {ws}.")

        
        if len(all_models) == 1:
            model_idx = 0
        else:
            loss_metrics = []
            for model_p in all_models:
                niter = _tmpf(model_p)
                win_D_diff = np.abs(self.D_losses[(niter-ws):(niter+ws)] + 2*np.log(0.5))
                win_G_diff = np.abs(self.G_losses[(niter-ws):(niter+ws)] + np.log(0.5))
                loss_metrics.append((win_D_diff.mean(), win_G_diff.mean(), win_D_diff.std(), win_G_diff.std()))
            loss_metrics = np.array(loss_metrics)
            kpidxs_m = np.bitwise_and(loss_metrics[:, 0]<m_tol, loss_metrics[:, 1]<m_tol)
            kpidxs_sd = np.bitwise_and(loss_metrics[:, 2]<sd_tol, loss_metrics[:, 3]<sd_tol)
            kpidxs_both = np.bitwise_and(kpidxs_m, kpidxs_sd)
            if kpidxs_both.sum() == 0:
                model_idx = -1
                logger.warning(f"Be careful, no trained model reaches convergence!")
            else:
                model_idx = np.where(kpidxs_both)[0][-1]
        
        logger.info(f"We load model {all_models[model_idx]}")
        model = self.Generator(self.d, self.nz);
        model.load_state_dict(torch.load(all_models[model_idx]))
        model.eval()
        return model
        