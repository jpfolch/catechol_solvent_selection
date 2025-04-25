

import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.functional import silu
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=5)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, latent_dnamics_dim=4, nhidden=20):
        self.latent_dim = latent_dim
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim + latent_dnamics_dim + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        zd  = x[:, latent_dim:]
        # augment t as input
        expand_t = t.unsqueeze(0).unsqueeze(-1)
        out = self.fc1(torch.concatenate([expand_t, x], axis=-1))
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return torch.concatenate([out, zd], axis=-1)


class LatentInitRecognition(nn.Module):
    def __init__(self, latent_init_dim, latent_dim, obs_dim, std_lower_bound=1E-2):
        super(LatentInitRecognition, self).__init__()
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(obs_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.mu = nn.Sequential(nn.Linear(latent_dim, latent_dim), self.silu, nn.Linear(latent_dim, latent_init_dim))
        self.sigma_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim), self.silu, nn.Linear(latent_dim, latent_init_dim))
        self._std_lower_bound = std_lower_bound
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.silu(out)
        out = self.fc2(out)
        out = self.silu(out)
        out = self.fc3(out)
        mu = self.mu(out)
        sigma = self._std_lower_bound + 0.9 *nn.functional.sigmoid(self.sigma_layer(out))
        return mu, sigma


class DynamicsRecognition(nn.Module):
    def __init__(self, latent_dynamics_dim, latent_dim, obs_dim, std_lower_bound=1E-2):
        super(DynamicsRecognition, self).__init__()
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(obs_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.mu = nn.Sequential(nn.Linear(latent_dim, latent_dim), self.silu, nn.Linear(latent_dim, latent_dynamics_dim))
        self.sigma_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim), self.silu, nn.Linear(latent_dim, latent_dynamics_dim))
        self._std_lower_bound = std_lower_bound
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.silu(out)
        out = self.fc2(out)
        out = self.silu(out)
        out = self.fc3(out)
        mu = self.mu(out)
        sigma = self._std_lower_bound + 0.9 *nn.functional.sigmoid(self.sigma_layer(out))
        return mu, sigma


class HeteroscedasticDecoder(nn.Module):
    def __init__(self, latent_dim, nhidden, obs_dim, std_lower_bound = 1e-1):
        super(HeteroscedasticDecoder, self).__init__()
        self.hidden_to_mu = nn.Sequential(nn.Linear(latent_dim, nhidden), nn.SiLU(), nn.Linear(nhidden, obs_dim))
        self.hidden_to_sigma = nn.Sequential(nn.Linear(latent_dim, nhidden), nn.SiLU(), nn.Linear(nhidden, obs_dim))
        self.tlz_to_hidden = nn.Sequential(nn.Linear(latent_dim * 3 + 1, nhidden), nn.SiLU(), nn.Linear(nhidden, nhidden), nn.SiLU(), nn.Linear(nhidden, latent_dim))
        self._std_lower_bound = std_lower_bound
    def __call__(self, target_t, z_t, z_0, z_d):
        """
        target_t: [timesteps]
        z_t: [sample_size, timesteps, D_dim]
        z_0: [sample_size, D_dim]
        z_d: [sample_size, D_dim]

        return [sample_size, time_steps, x_dim]
        """
        # concatenation target_t, z_d, z_t, z_0
        expand_target_t = target_t.unsqueeze(0).unsqueeze(-1)
        expand_z_d = z_d.unsqueeze(1).expand(-1, z_t.size(1), -1)
        expand_z_0 = z_0.unsqueeze(1).expand(-1, z_t.size(1), -1)
        hidden = self.tlz_to_hidden(\
            torch.concatenate([expand_target_t, expand_z_d, z_t, expand_z_0], axis=-1))

        mu = self.hidden_to_mu(hidden)
        sigma = self.hidden_to_sigma(hidden)
        sigma = self._std_lower_bound + 0.9 * nn.functional.softplus(sigma)
        # var = torch.exp(log_var) + self._var_lower_bound
        return mu, sigma



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



# def log_normal_pdf(x, mean, logvar):
#     const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
#     const = torch.log(const)
#     return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


# def normal_kl(mu1, lv1, mu2, lv2):
#     v1 = torch.exp(lv1)
#     v2 = torch.exp(lv2)
#     lstd1 = lv1 / 2.
#     lstd2 = lv2 / 2.
# 
#     kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
#     return kl

def normal_kl(mu1, var1, mu2, var2):
    """
    Computes the KL divergence between two diagonal Gaussians:
    q ~ N(mu1, var1), p ~ N(mu2, var2)
    
    Inputs:
        mu1: Tensor of shape [..., D]
        var1: Tensor of shape [..., D]
        mu2: Tensor of shape [..., D]
        var2: Tensor of shape [..., D]
        
    Returns:
        kl: Tensor of shape [...] (sum over D)
    """
    kl = 0.5 * ( (var1 / var2) + ((mu2 - mu1)**2) / var2 - 1 + torch.log(var2 / var1) )
    return kl.sum(-1)



if __name__ == '__main__':
    latent_dim = 32
    latent_dynamics_dim = 32
    n_ode_hidden = 64
    n_l_in_rec_hidden = 64
    n_l_d_rec_hidden = 64
    n_dec_hidden = 64
    obs_dim = 3
    start = 0.
    stop = 1.0
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # model
    func = LatentODEfunc(latent_dim, latent_dynamics_dim, n_ode_hidden).to(device)
    l_in_rec = LatentInitRecognition(latent_dim, n_l_in_rec_hidden, 5 + 1).to(device)
    l_d_rec = DynamicsRecognition(latent_dim, n_l_d_rec_hidden, 5 + 1).to(device)
    dec = HeteroscedasticDecoder(latent_dim, n_dec_hidden, obs_dim).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(l_in_rec.parameters()) + list(l_d_rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()


    # TODO: Change data input format
    from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
    from catechol.data.loader import load_single_solvent_data, train_test_split
    train_X, train_Y = load_single_solvent_data()
    
    # remove unnecessary columns
    train_X = train_X[INPUT_LABELS_SINGLE_SOLVENT]
    from catechol.data.featurizations import featurize_input_df
    original_solvent_name = train_X["SOLVENT NAME"].copy()
    train_X = featurize_input_df(train_X, featurization="acs_pca_descriptors")
    train_X['SOLVENT EMBEDDING'] = train_X[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].values.tolist()
    train_X['SOLVENT NAME'] = original_solvent_name
    train_X = train_X.drop(columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    
    train_X, test_X = train_test_split(train_X, train_percentage=0.8, seed=1)
    train_Y, test_Y = train_test_split(train_Y, train_percentage=0.8, seed=1)

    def _reformulate_data_as_time_series(
        x: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[list[tuple[str, float]], list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Reformulate the data as a list of time series grouped by (SOLVENT NAME, Temperature).

        Returns:
            keys: list of (solvent name, temperature)
            x_series: list of pd.DataFrame, each containing a trajectory sorted by Residence Time
            y_series: list of pd.DataFrame, aligned with x_series
        """
        # Combine x and y so they can be grouped and sorted together
        combined = pd.concat([x, y], axis=1)

        # Group by (SOLVENT NAME, Temperature)
        grouped = combined.groupby(["SOLVENT NAME", "Temperature"])

        keys = []
        x_series = []
        y_series = []

        for (solvent, temp), group in grouped:
            group = group.sort_values(by="Residence Time").reset_index(drop=True)
            keys.append((solvent, float(temp)))
            x_series.append(group[["Residence Time"]])  # or other x columns if needed
            y_series.append(group[y.columns])  # retain all original y columns

        return keys, x_series, y_series


    def unique_with_tolerance(x, tol=1e-6):
        x = np.sort(x)
        unique = [x[0]]
        idx = [0]
        for i in range(1, len(x)):
            if np.abs(x[i] - unique[-1]) > tol:
                unique.append(x[i])
            idx.append(len(unique) - 1)
        return np.array(unique), np.array(idx)

    
    keys, traj_inputs, traj_outputs = _reformulate_data_as_time_series(train_X, train_Y)

    # TODO: 
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            l_in_rec.load_state_dict(checkpoint['l_in_rec_state_dict'])
            l_d_rec.load_state_dict(checkpoint['l_d_rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in tqdm(np.arange(1, args.niters + 1)):
            optimizer.zero_grad()
            for (solvent_key, temp), traj_in, traj_o in tqdm(zip(keys, traj_inputs, traj_outputs)):
                measurement_time = np.squeeze(np.asarray(traj_in).T)
                measurement_time, inverse_indices = unique_with_tolerance(np.atleast_1d(measurement_time)) # np.unique(measurement_time, return_inverse=True)
                inverse_indices = np.atleast_1d(inverse_indices) # it could happen there is no duplication, here this is needed to make sure inverse_indices not reduce dim laterwards
                solvent_emb = train_X.loc[train_X['SOLVENT NAME'] == solvent_key, 'SOLVENT EMBEDDING'].values[0]
                solvent_emb_temperature = torch.Tensor(np.atleast_2d(np.concatenate([solvent_emb, np.atleast_1d(temp)], axis=-1))).to(device)
                # infer and sample latent initial state
                qz0_mean, qz0_std = l_in_rec.forward(solvent_emb_temperature)
                epsilon_z0 = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon_z0 * qz0_std + qz0_mean
                # infer and sample latent dynamics
                qzd_mean, qzd_std = l_d_rec.forward(solvent_emb_temperature)
                epsilon_zd = torch.randn(qzd_mean.size()).to(device)
                zd = epsilon_zd * qzd_std + qzd_mean

                # forward in time and solve ode for reconstructions
                # TODO: How to feed zd as a control variable?
                try:
                    pred_z = odeint(func, 
                                    torch.concat([z0, zd], axis=-1), \
                                    torch.Tensor(measurement_time).to(device)).permute(1, 0, 2)[:, :, :latent_dim]
                except:
                    print(f'measurement_time: {measurement_time}')
                # pred_z = pred_z[inverse_indices].permute(1, 0, 2)
                pred_mean, pred_std = dec(torch.Tensor(measurement_time).to(device), pred_z, z0, zd)
                pred_mean = pred_mean.permute(1, 0, 2)[inverse_indices].permute(1, 0, 2)
                pred_std = pred_std.permute(1, 0, 2)[inverse_indices].permute(1, 0, 2)
                # compute loss
                logpx = torch.distributions.normal.Normal(loc = pred_mean[0], scale = pred_std[0]).log_prob(torch.Tensor(np.asarray(traj_o)).to(device)).sum(-1).mean(-1) # .sum(-1)
                # noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
                # noise_logvar = 2. * torch.log(noise_std_).to(device)
                # logpx = log_normal_pdf(
                #     samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
                # calculate log pdf, implement our own
                # TODO: Implement the KL divergence
                qz0_logvar = torch.log(qz0_std ** 2)
                qzd_logvar = torch.log(qzd_std ** 2)
                pz0_mean = torch.zeros_like(qz0_mean).to(device)
                pz0_logvar = torch.zeros_like(qz0_logvar).to(device)
                pzd_mean  = torch.zeros_like(qzd_mean).to(device)
                pzd_logvar = torch.zeros_like(qzd_logvar).to(device)
                analytic_kl_qz0 = normal_kl(qz0_mean, qz0_std ** 2, pz0_mean, torch.ones_like(qz0_std)).sum(-1)
                analytic_kl_qzd = normal_kl(qz0_mean, qzd_std ** 2, pz0_mean, torch.ones_like(qzd_std)).sum(-1)
                loss = torch.mean(-logpx + analytic_kl_qz0 + analytic_kl_qzd, dim=0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                loss_meter.update(loss.item())

                # loss = torch.mean(-logpx + analytic_kl_qz0 + analytic_kl_qzd, dim=0)
            # print('Iter: {}, logpx: {:.4f},  kl_qz0: {:.4f}, kl_qzd: {:.4f}, running avg elbo: {:.4f}'.format(itr, logpx, float(analytic_kl_qz0), float(analytic_kl_qzd), -loss_meter.avg))
            if itr % args.save_freq == 0:
                ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'l_in_rec_state_dict': l_in_rec.state_dict(),
                    'l_d_rec_state_dict': l_d_rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            
            print('Iter: {},  running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'l_in_rec_state_dict': l_in_rec.state_dict(),
                'l_d_rec_state_dict': l_d_rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.visualize:
        with torch.no_grad():
            pass

        plt.figure()
        plt.legend()
        plt.savefig('./vis.png', dpi=500)
        print('Saved visualization figure at {}'.format('./vis.png'))