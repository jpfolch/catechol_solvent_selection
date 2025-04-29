import pandas as pd
import os
import torch
from torch import nn
import torch.optim as optim

import random
import numpy as np
from catechol import metrics
from tqdm import tqdm
from torchdiffeq import odeint
from catechol.data.data_labels import get_data_labels_mean_var
from catechol.data.featurizations import FeaturizationType, featurize_input_df
from catechol.data.loader import train_test_split
from .base_model import Model


class LantentODE(Model):
    """
    A Variational Inference based latent ODE
    """

    normalize_inputs = False

    def __init__(
        self,
        device,
        featurization_dim: int,
        state_dim: int = 3,
        latent_state_dim: int = 32,
        latent_dynamics_dim: int = 32,
        h_dim_ode: int = 64,
        h_dim_x0: int = 64,
        h_dim_dynmcs: int = 64,
        h_dim_dec: int = 64,
        featurization: FeaturizationType | None = None,
    ):
        super().__init__(featurization=featurization)
        self.device = device
        self.latent_state_dim = latent_state_dim
        self.func = _LatentODEfunc(latent_state_dim, latent_dynamics_dim, h_dim_ode).to(
            device
        )
        self.l_in_rec = _LatentInitRecognition(
            latent_state_dim, h_dim_x0, featurization_dim + 1
        ).to(device)
        self.l_d_rec = _DynamicsRecognition(
            latent_dynamics_dim, h_dim_dynmcs, featurization_dim + 1
        ).to(device)
        self.dec = _HeteroscedasticDecoder(latent_state_dim, h_dim_dec, state_dim, latent_dynamics_dim).to(
            device
        )

    def _train(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        learning_rate: float,
        train_epoch: int,
        use_pretrained_model: bool = False,
        train_dir: str = None,
        kl_weight: float = 1.0,
        mc_sample_num: int = 1,
        validation_fraction: float = 0.0,
        **kwargs,
    ) -> None:

        # initialize
        self.func.to(self.device)
        self.l_in_rec.to(self.device)
        self.l_d_rec.to(self.device)
        self.dec.to(self.device)

        params = (
            list(self.func.parameters())
            + list(self.dec.parameters())
            + list(self.l_in_rec.parameters())
            + list(self.l_d_rec.parameters())
        )
        optimizer = optim.Adam(params, lr=learning_rate)

        # data preprocessing: process the data to trajectories
        # TODO: Hard code atm
        original_solvent_name = train_X["SOLVENT NAME"].copy()
        train_X = featurize_input_df(train_X, featurization="acs_pca_descriptors")
        train_X["SOLVENT EMBEDDING"] = train_X[
            ["PC1", "PC2", "PC3", "PC4", "PC5"]
        ].values.tolist()
        train_X["SOLVENT NAME"] = original_solvent_name
        train_X = train_X.drop(columns=["PC1", "PC2", "PC3", "PC4", "PC5"])

        if validation_fraction != 0.0:  # validation fraction enabled
            train_X, val_X = train_test_split(train_X, train_percentage=0.9, seed=1)
            train_Y, val_Y = train_test_split(train_Y, train_percentage=0.9, seed=1)

        keys, traj_inputs, traj_outputs = _formulate_data_as_time_series(
            train_X, train_Y
        )

        latest_ckpt, latest_ckpt_path = find_latest_ckpt(train_dir)
        if latest_ckpt_path is not None:
            checkpoint = torch.load(latest_ckpt_path)
            self.func.load_state_dict(checkpoint["func_state_dict"])
            self.l_in_rec.load_state_dict(checkpoint["l_in_rec_state_dict"])
            self.l_d_rec.load_state_dict(checkpoint["l_d_rec_state_dict"])
            self.dec.load_state_dict(checkpoint["dec_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded ckpt from {}".format(latest_ckpt_path))
        else:
            latest_ckpt = 0
            print('No existing ckpt found')

        if use_pretrained_model:
            pass  # no need to retrain
        else:
            pbar = tqdm(np.arange(latest_ckpt, train_epoch + 1))
            for itr in pbar:
                optimizer.zero_grad()
                total_logpx, total_kl_z0, total_kl_zd, total_mse = 0.0, 0.0, 0.0, 0.0
                count = 0
                # shuffle data
                data = list(zip(keys, traj_inputs, traj_outputs))
                random.shuffle(data)
                for (
                    (solvent_key, temp),
                    traj_in,
                    traj_o,
                ) in data:  # tqdm(zip(keys, traj_inputs, traj_outputs)):
                    measurement_time = torch.atleast_1d(
                        torch.Tensor(traj_in.to_numpy().T).squeeze()
                    ).to(self.device)
                    # exclude (near) duplicate time to allow odeint solve
                    measurement_time, inverse_indices = _unique_with_tolerance(
                        measurement_time
                    )
                    # get the embedding(featurization) of solvent
                    solvent_emb = train_X.loc[
                        train_X["SOLVENT NAME"] == solvent_key, "SOLVENT EMBEDDING"
                    ].iloc[0]
                    # note: only single traj here
                    solvent_emb_temperature = (
                        torch.cat(
                            [torch.Tensor(solvent_emb), torch.Tensor([temp])], dim=-1
                        )
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    # infer and sample latent initial state
                    qz0_mean, qz0_std = self.l_in_rec.forward(solvent_emb_temperature)
                    # # infer and sample latent dynamics
                    qzd_mean, qzd_std = self.l_d_rec.forward(solvent_emb_temperature)

                    # Sample MC samples in a batch
                    epsilon_z0 = torch.randn(mc_sample_num, *qz0_mean.size()).to(
                        self.device
                    )
                    z0 = (epsilon_z0 * qz0_std + qz0_mean).permute(
                        1, 0, 2
                    )  # [batch_dim, mc_sample_num, latent_dim]

                    epsilon_zd = torch.randn(mc_sample_num, *qzd_mean.size()).to(
                        self.device
                    )
                    zd = (epsilon_zd * qzd_std + qzd_mean).permute(
                        1, 0, 2
                    )  # [batch_dim, mc_sample_num, latent_dim]

                    # forward in time and solve ode for reconstructions
                    pred_z = odeint(
                        self.func,
                        torch.concat([z0, zd], axis=-1),
                        torch.Tensor(measurement_time).to(self.device),
                    ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

                    pred_mean, pred_std = self.dec(measurement_time, pred_z, z0, zd)
                    pred_mean = pred_mean.permute(2, 0, 1, 3)[inverse_indices].permute(
                        1, 2, 0, 3
                    )
                    pred_std = pred_std.permute(2, 0, 1, 3)[inverse_indices].permute(
                        1, 2, 0, 3
                    )
                    # compute ELBO loss
                    logpx = (
                        torch.distributions.normal.Normal(
                            loc=pred_mean[0], scale=pred_std[0]
                        )
                        .log_prob(torch.Tensor(traj_o.to_numpy()).to(self.device))
                        .sum(-1)
                        .sum(-1)
                        .mean()
                    )

                    pz0_mean = torch.zeros_like(qz0_mean).to(self.device)
                    pzd_mean = torch.zeros_like(qzd_mean).to(self.device)
                    analytic_kl_qz0 = _normal_kl(
                        qz0_mean, qz0_std**2, pz0_mean, torch.ones_like(qz0_std)
                    ).sum(-1)
                    analytic_kl_qzd = _normal_kl(
                        qzd_mean, qzd_std**2, pzd_mean, torch.ones_like(qzd_std)
                    ).sum(-1)
                    loss = torch.mean(
                        -logpx + kl_weight * (analytic_kl_qz0 + analytic_kl_qzd), dim=0
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        params, max_norm=kwargs.get("grad_clip_max_norm", 2.0)
                    )
                    optimizer.step()

                    mse_x = (
                        (pred_mean[0] - torch.Tensor(traj_o.to_numpy()).to(self.device))
                        ** 2
                    ).mean()
                    total_mse += mse_x.detach().sum().item()
                    total_logpx += logpx.detach().sum().item()
                    total_kl_z0 += analytic_kl_qz0.detach().sum().item()
                    total_kl_zd += analytic_kl_qzd.detach().sum().item()
                    count += 1

                # pred val
                if validation_fraction != 0.0:
                    _predictions = self._predict(val_X, mc_sample_num=32)
                    val_mse = metrics.mse(_predictions, val_Y)
                else:
                    val_mse = np.inf

                pbar.set_postfix(
                    {
                        "avg_logpx": total_logpx / count,
                        "avg_kl_z0": total_kl_z0 / count,
                        "avg_kl_zd": total_kl_zd / count,
                        "loss_mse": total_mse / count,
                        "val_mse": "{:.4f}".format(val_mse),
                    }
                )

                if itr % kwargs.get("save_freq", torch.inf) == 0:
                    os.makedirs(train_dir, exist_ok=True)
                    ckpt_path = os.path.join(train_dir, f"ckpt_epoch_{itr}.pth")
                    torch.save({
                        "func_state_dict": self.func.state_dict(),
                        "l_in_rec_state_dict": self.l_in_rec.state_dict(),
                        "l_d_rec_state_dict": self.l_d_rec.state_dict(),
                        "dec_state_dict": self.dec.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"Stored ckpt at {ckpt_path}")

    def _predict(self, test_X: pd.DataFrame, mc_sample_num: int = 1) -> pd.DataFrame:
        original_solvent_name = test_X["SOLVENT NAME"].copy()
        test_X = featurize_input_df(test_X, featurization="acs_pca_descriptors")
        test_X["SOLVENT EMBEDDING"] = test_X[
            ["PC1", "PC2", "PC3", "PC4", "PC5"]
        ].values.tolist()
        test_X["SOLVENT NAME"] = original_solvent_name
        test_X = test_X.drop(columns=["PC1", "PC2", "PC3", "PC4", "PC5"])

        keys, traj_inputs = _formulate_data_as_time_series(test_X)
        with torch.no_grad():
            pred_means = []
            pred_vars = []
            for (solvent_key, temp), traj_in in tqdm(zip(keys, traj_inputs)):
                measurement_time = torch.atleast_1d(
                    torch.Tensor(traj_in.to_numpy().T).squeeze()
                ).to(self.device)
                # exclude (near) duplicate time to allow odeint solve
                measurement_time, inverse_indices = _unique_with_tolerance(
                    measurement_time
                )
                # get the embedding(featurization) of solvent
                solvent_emb = test_X.loc[
                    test_X["SOLVENT NAME"] == solvent_key, "SOLVENT EMBEDDING"
                ].iloc[0]
                # note: only single traj here
                solvent_emb_temperature = (
                    torch.cat([torch.Tensor(solvent_emb), torch.Tensor([temp])], dim=-1)
                    .unsqueeze(0)
                    .to(self.device)
                )

                qz0_mean, qz0_std = self.l_in_rec.forward(solvent_emb_temperature)
                # # infer and sample latent dynamics
                qzd_mean, qzd_std = self.l_d_rec.forward(solvent_emb_temperature)

                # Sample MC samples in a batch
                epsilon_z0 = torch.randn(mc_sample_num, *qz0_mean.size()).to(
                    self.device
                )
                z0 = (epsilon_z0 * qz0_std + qz0_mean).permute(
                    1, 0, 2
                )  # [batch_dim, mc_sample_num, latent_dim]

                epsilon_zd = torch.randn(mc_sample_num, *qzd_mean.size()).to(
                    self.device
                )
                zd = (epsilon_zd * qzd_std + qzd_mean).permute(
                    1, 0, 2
                )  # [batch_dim, mc_sample_num, latent_dim]

                # forward in time and solve ode for reconstructions
                pred_z = odeint(
                    self.func,
                    torch.concat([z0, zd], axis=-1),
                    torch.Tensor(measurement_time).to(self.device),
                ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

                pred_mean, pred_std = self.dec(measurement_time, pred_z, z0, zd)
                pred_mean = pred_mean.permute(2, 0, 1, 3)[inverse_indices].permute(
                    1, 2, 0, 3
                )
                pred_std = pred_std.permute(2, 0, 1, 3)[inverse_indices].permute(
                    1, 2, 0, 3
                )
                pred_means.append(pred_mean[0].mean(0))  # average cross mc dim
                pred_vars.append((pred_std**2)[0].mean(0))  # average cross mc dim

        mean, var = _formulate_time_series_as_data(
            keys, traj_inputs, pred_means, pred_vars
        )
        mean_lbl, var_lbl = get_data_labels_mean_var()
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass


class ExplicitODE(Model):
    """
    A Variational Inference based ODE
    """

    normalize_inputs = False

    def __init__(
        self,
        device,
        featurization_dim: int,
        state_dim: int = 3,
        dynamics_dim: int = 32,
        h_dim_ode: int = 64,
        h_dim_dynmcs: int = 64,
        h_dim_dec: int = 64,
        featurization: FeaturizationType | None = None,
    ):
        super().__init__(featurization=featurization)
        self.device = device
        self.latent_state_dim = state_dim
        self.func = _LatentODEfunc(state_dim, dynamics_dim, h_dim_ode).to(
            device
        )
        self.l_d_rec = _DynamicsRecognition(
            dynamics_dim, h_dim_dynmcs, featurization_dim + 1
        ).to(device)
        self.dec = _HeteroscedasticDecoder(state_dim, h_dim_dec, state_dim, dynamics_dim).to(
            device
        )

    def _train(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        learning_rate: float,
        train_epoch: int,
        use_pretrained_model: bool = False,
        train_dir: str = None,
        kl_weight: float = 1.0,
        mc_sample_num: int = 1,
        validation_fraction: float = 0.0,
        **kwargs,
    ) -> None:

        # initialize
        self.func.to(self.device)
        self.l_d_rec.to(self.device)
        self.dec.to(self.device)

        params = (
            list(self.func.parameters())
            + list(self.dec.parameters())
            + list(self.l_d_rec.parameters())
        )
        optimizer = optim.Adam(params, lr=learning_rate)

        # data preprocessing: process the data to trajectories
        # TODO: Hard code atm
        original_solvent_name = train_X["SOLVENT NAME"].copy()
        train_X = featurize_input_df(train_X, featurization="acs_pca_descriptors")
        train_X["SOLVENT EMBEDDING"] = train_X[
            ["PC1", "PC2", "PC3", "PC4", "PC5"]
        ].values.tolist()
        train_X["SOLVENT NAME"] = original_solvent_name
        train_X = train_X.drop(columns=["PC1", "PC2", "PC3", "PC4", "PC5"])

        if validation_fraction != 0.0:  # validation fraction enabled
            train_X, val_X = train_test_split(train_X, train_percentage=0.9, seed=1)
            train_Y, val_Y = train_test_split(train_Y, train_percentage=0.9, seed=1)

        keys, traj_inputs, traj_outputs = _formulate_data_as_time_series(
            train_X, train_Y
        )

        latest_ckpt, latest_ckpt_path = find_latest_ckpt(train_dir)
        if latest_ckpt_path is not None:
            checkpoint = torch.load(latest_ckpt_path)
            self.func.load_state_dict(checkpoint["func_state_dict"])
            self.l_d_rec.load_state_dict(checkpoint["l_d_rec_state_dict"])
            self.dec.load_state_dict(checkpoint["dec_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded ckpt from {}".format(latest_ckpt_path))
        else:
            latest_ckpt = 0
            print('No existing ckpt found')

        if use_pretrained_model:
            pass  # no need to retrain
        else:
            pbar = tqdm(np.arange(latest_ckpt, train_epoch + 1))
            for itr in pbar:
                optimizer.zero_grad()
                total_logpx, total_kl_zd, total_mse = 0.0, 0.0, 0.0
                count = 0
                # shuffle data
                data = list(zip(keys, traj_inputs, traj_outputs))
                random.shuffle(data)
                for (
                    (solvent_key, temp),
                    traj_in,
                    traj_o,
                ) in data:  # tqdm(zip(keys, traj_inputs, traj_outputs)):
                    measurement_time = torch.atleast_1d(
                        torch.Tensor(traj_in.to_numpy().T).squeeze()
                    ).to(self.device)
                    # exclude (near) duplicate time to allow odeint solve
                    measurement_time, inverse_indices = _unique_with_tolerance(
                        measurement_time
                    )
                    # get the embedding(featurization) of solvent
                    solvent_emb = train_X.loc[
                        train_X["SOLVENT NAME"] == solvent_key, "SOLVENT EMBEDDING"
                    ].iloc[0]
                    # note: only single traj here
                    solvent_emb_temperature = (
                        torch.cat(
                            [torch.Tensor(solvent_emb), torch.Tensor([temp])], dim=-1
                        )
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    # infer and sample latent dynamics
                    qzd_mean, qzd_std = self.l_d_rec.forward(solvent_emb_temperature)

                    # Sample MC samples in a batch
                    z0 = torch.Tensor([[[1.0, 0.0, 0.0]]]).repeat(1, mc_sample_num, 1).to(qzd_mean.device) # [batch_dim, mc_sample_num, latent_dim]

                    epsilon_zd = torch.randn(mc_sample_num, *qzd_mean.size()).to(
                        self.device
                    )
                    zd = (epsilon_zd * qzd_std + qzd_mean).permute(
                        1, 0, 2
                    )  # [batch_dim, mc_sample_num, latent_dim]

                    # forward in time and solve ode for reconstructions
                    pred_z = odeint(
                        self.func,
                        torch.concat([z0, zd], axis=-1),
                        torch.Tensor(measurement_time).to(self.device),
                    ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

                    pred_mean, pred_std = self.dec(measurement_time, pred_z, z0, zd)
                    pred_mean = pred_mean.permute(2, 0, 1, 3)[inverse_indices].permute(
                        1, 2, 0, 3
                    )
                    pred_std = pred_std.permute(2, 0, 1, 3)[inverse_indices].permute(
                        1, 2, 0, 3
                    )
                    # compute ELBO loss
                    logpx = (
                        torch.distributions.normal.Normal(
                            loc=pred_mean[0], scale=pred_std[0]
                        )
                        .log_prob(torch.Tensor(traj_o.to_numpy()).to(self.device))
                        .sum(-1)
                        .sum(-1)
                        .mean()
                    )

                    pzd_mean = torch.zeros_like(qzd_mean).to(self.device)
                    analytic_kl_qzd = _normal_kl(
                        qzd_mean, qzd_std**2, pzd_mean, torch.ones_like(qzd_std)
                    ).sum(-1)
                    loss = torch.mean(
                        -logpx + kl_weight * (analytic_kl_qzd), dim=0
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        params, max_norm=kwargs.get("grad_clip_max_norm", 2.0)
                    )
                    optimizer.step()

                    mse_x = (
                        (pred_mean[0] - torch.Tensor(traj_o.to_numpy()).to(self.device))
                        ** 2
                    ).mean()
                    total_mse += mse_x.detach().sum().item()
                    total_logpx += logpx.detach().sum().item()
                    total_kl_zd += analytic_kl_qzd.detach().sum().item()
                    count += 1

                # pred val
                if validation_fraction != 0.0:
                    _predictions = self._predict(val_X, mc_sample_num=32)
                    val_mse = metrics.mse(_predictions, val_Y)
                else:
                    val_mse = np.inf

                pbar.set_postfix(
                    {
                        "avg_logpx": total_logpx / count,
                        "avg_kl_zd": total_kl_zd / count,
                        "loss_mse": total_mse / count,
                        "val_mse": "{:.4f}".format(val_mse),
                    }
                )

                if itr % kwargs.get("save_freq", torch.inf) == 0:
                    os.makedirs(train_dir, exist_ok=True)
                    ckpt_path = os.path.join(train_dir, f"ckpt_epoch_{itr}.pth")
                    torch.save({
                        "func_state_dict": self.func.state_dict(),
                        "l_d_rec_state_dict": self.l_d_rec.state_dict(),
                        "dec_state_dict": self.dec.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"Stored ckpt at {ckpt_path}")

    def _predict(self, test_X: pd.DataFrame, mc_sample_num: int = 1) -> pd.DataFrame:
        original_solvent_name = test_X["SOLVENT NAME"].copy()
        test_X = featurize_input_df(test_X, featurization="acs_pca_descriptors")
        test_X["SOLVENT EMBEDDING"] = test_X[
            ["PC1", "PC2", "PC3", "PC4", "PC5"]
        ].values.tolist()
        test_X["SOLVENT NAME"] = original_solvent_name
        test_X = test_X.drop(columns=["PC1", "PC2", "PC3", "PC4", "PC5"])

        keys, traj_inputs = _formulate_data_as_time_series(test_X)
        with torch.no_grad():
            pred_means = []
            pred_vars = []
            for (solvent_key, temp), traj_in in tqdm(zip(keys, traj_inputs)):
                measurement_time = torch.atleast_1d(
                    torch.Tensor(traj_in.to_numpy().T).squeeze()
                ).to(self.device)
                # exclude (near) duplicate time to allow odeint solve
                measurement_time, inverse_indices = _unique_with_tolerance(
                    measurement_time
                )
                # get the embedding(featurization) of solvent
                solvent_emb = test_X.loc[
                    test_X["SOLVENT NAME"] == solvent_key, "SOLVENT EMBEDDING"
                ].iloc[0]
                # note: only single traj here
                solvent_emb_temperature = (
                    torch.cat([torch.Tensor(solvent_emb), torch.Tensor([temp])], dim=-1)
                    .unsqueeze(0)
                    .to(self.device)
                )

                z0 = torch.Tensor([[[1.0, 0.0, 0.0]]]).repeat(1, mc_sample_num, 1).to(solvent_emb_temperature.device) # [batch_dim, mc_sample_num, latent_dim]

                # # infer and sample latent dynamics
                qzd_mean, qzd_std = self.l_d_rec.forward(solvent_emb_temperature)

                # Sample MC samples in a batch
                epsilon_zd = torch.randn(mc_sample_num, *qzd_mean.size()).to(
                    self.device
                )
                zd = (epsilon_zd * qzd_std + qzd_mean).permute(
                    1, 0, 2
                )  # [batch_dim, mc_sample_num, latent_dim]

                # forward in time and solve ode for reconstructions
                pred_z = odeint(
                    self.func,
                    torch.concat([z0, zd], axis=-1),
                    torch.Tensor(measurement_time).to(self.device),
                ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

                pred_mean, pred_std = self.dec(measurement_time, pred_z, z0, zd)
                pred_mean = pred_mean.permute(2, 0, 1, 3)[inverse_indices].permute(
                    1, 2, 0, 3
                )
                pred_std = pred_std.permute(2, 0, 1, 3)[inverse_indices].permute(
                    1, 2, 0, 3
                )
                pred_means.append(pred_mean[0].mean(0))  # average cross mc dim
                pred_vars.append((pred_std**2)[0].mean(0))  # average cross mc dim

        mean, var = _formulate_time_series_as_data(
            keys, traj_inputs, pred_means, pred_vars
        )
        mean_lbl, var_lbl = get_data_labels_mean_var()
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass



class _LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, latent_dynamics_dim=4, nhidden=20):
        self.latent_dynamics_dim = latent_dynamics_dim
        super(_LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim + latent_dynamics_dim + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        zd = x[..., - self.latent_dynamics_dim :]
        # augment t as input
        expand_t = torch.ones(zd.shape[:-1]).unsqueeze(-1).to(x.device) * t
        out = self.fc1(torch.concatenate([expand_t, x], axis=-1))
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return torch.concatenate([out, zd], axis=-1)


class _LatentInitRecognition(nn.Module):
    def __init__(self, latent_init_dim, latent_dim, obs_dim, std_lower_bound=1e-2):
        super(_LatentInitRecognition, self).__init__()
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(obs_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.silu,
            nn.Linear(latent_dim, latent_init_dim),
        )
        self.sigma_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.silu,
            nn.Linear(latent_dim, latent_init_dim),
        )
        self._std_lower_bound = std_lower_bound

    def forward(self, x):
        out = self.fc1(x)
        out = self.silu(out)
        out = self.fc2(out)
        out = self.silu(out)
        out = self.fc3(out)
        mu = self.mu(out)
        sigma = self._std_lower_bound + 0.9 * nn.functional.softplus(
            self.sigma_layer(out)
        )
        return mu, sigma


class _DynamicsRecognition(nn.Module):
    def __init__(self, latent_dynamics_dim, latent_dim, obs_dim, std_lower_bound=1e-2):
        super(_DynamicsRecognition, self).__init__()
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(obs_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.silu,
            nn.Linear(latent_dim, latent_dynamics_dim),
        )
        self.sigma_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.silu,
            nn.Linear(latent_dim, latent_dynamics_dim),
        )
        self._std_lower_bound = std_lower_bound

    def forward(self, x):
        out = self.fc1(x)
        out = self.silu(out)
        out = self.fc2(out)
        out = self.silu(out)
        out = self.fc3(out)
        mu = self.mu(out)
        sigma = self._std_lower_bound + 0.9 * nn.functional.softplus(
            self.sigma_layer(out)
        )
        return mu, sigma


class _HeteroscedasticDecoder(nn.Module):
    def __init__(self, latent_state_dim, nhidden, obs_dim, dynamics_dim, std_lower_bound=1e-2):
        super(_HeteroscedasticDecoder, self).__init__()
        self.hidden_to_mu = nn.Sequential(
            nn.Linear(nhidden, nhidden), nn.SiLU(), nn.Linear(nhidden, obs_dim)
        )
        self.hidden_to_sigma = nn.Sequential(
            nn.Linear(nhidden, nhidden), nn.SiLU(), nn.Linear(nhidden, obs_dim)
        )
        self.tlz_to_hidden = nn.Sequential(
            nn.Linear(latent_state_dim * 2 + dynamics_dim + 1, nhidden),
            nn.SiLU(),
            nn.Linear(nhidden, nhidden),
            nn.SiLU(),
            nn.Linear(nhidden, nhidden),
        )
        self._std_lower_bound = std_lower_bound

    def __call__(self, target_t, z_t, z_0, z_d):
        """
        target_t: [timesteps]
        z_t: [batch_size, mc_sample_size, timesteps, D_dim]
        z_0: [batch_size,  mc_sample_size, D_dim]
        z_d: [batch_size, mc_sample_size,  D_dim]

        return [sample_size, time_steps, x_dim]
        """
        # concatenation target_t, z_d, z_t, z_0
        expand_target_t = (
            torch.ones(z_t.shape[:-1]).to(target_t.device) * target_t
        ).unsqueeze(-1)
        expand_z_d = z_d.unsqueeze(-2).expand(-1, -1, z_t.size(-2), -1)
        expand_z_0 = z_0.unsqueeze(-2).expand(-1, -1, z_t.size(-2), -1)
        hidden = self.tlz_to_hidden(
            torch.concatenate([expand_target_t, expand_z_d, z_t, expand_z_0], axis=-1)
        )

        mu = self.hidden_to_mu(hidden)
        sigma = self.hidden_to_sigma(hidden)
        sigma = self._std_lower_bound + 0.9 * nn.functional.softplus(sigma)
        return mu, sigma


def _unique_with_tolerance(x, tol=1e-6):
    x, _ = torch.sort(x)
    unique = [x[0]]
    idx = [0]
    for i in range(1, len(x)):
        if torch.abs(x[i] - unique[-1]) > tol:
            unique.append(x[i])
        idx.append(len(unique) - 1)
    unique = torch.stack(unique)
    idx = torch.tensor(idx, dtype=torch.long)
    return unique, torch.atleast_1d(idx)


def _formulate_data_as_time_series(
    x: pd.DataFrame, y: pd.DataFrame | None = None
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
    y_series = None if y is None else []
    for (solvent, temp), group in grouped:
        group = group.sort_values(by="Residence Time").reset_index(drop=True)
        keys.append((solvent, float(temp)))
        x_series.append(group[["Residence Time"]])  # or other x columns if needed
        if y is not None:
            y_series.append(group[y.columns])  # retain all original y columns
    if y is not None:
        return keys, x_series, y_series
    else:
        return keys, x_series


def _formulate_time_series_as_data(
    keys: list[tuple[str, float]],
    x_series: list[pd.DataFrame],
    pred_means: list[torch.Tensor],
    pred_vars: list[torch.Tensor],
) -> tuple:
    """
    Reformulate a list of predicted time series back into a single data matrix.

    Args:
        keys: list of (solvent name, temperature)
        x_series: list of pd.DataFrame, each containing 'Residence Time' per trajectory
        pred_means: list of torch.Tensor, each [timesteps, state_dim]
        pred_vars: list of torch.Tensor, each [timesteps, state_dim]

    Returns:
        mean: [N, D] array
        var: [N, D] array
    """

    dfs_mean = []
    dfs_var = []

    for (solvent_name, temp), x_df, mean_tensor, var_tensor in zip(
        keys, x_series, pred_means, pred_vars
    ):
        n_time = x_df.shape[0]

        mean_np = mean_tensor.detach().cpu().numpy()  # [timesteps, D]
        var_np = var_tensor.detach().cpu().numpy()  # [timesteps, D]

        # Create a DataFrame for this trajectory
        traj_mean_df = pd.DataFrame(
            mean_np, columns=[f"state_{i}" for i in range(mean_np.shape[-1])]
        )
        traj_var_df = pd.DataFrame(
            var_np, columns=[f"state_{i}" for i in range(var_np.shape[-1])]
        )

        # Add metadata
        traj_mean_df["SOLVENT NAME"] = solvent_name
        traj_mean_df["Temperature"] = temp
        traj_mean_df["Residence Time"] = x_df["Residence Time"].values  # original times

        traj_var_df["SOLVENT NAME"] = solvent_name
        traj_var_df["Temperature"] = temp
        traj_var_df["Residence Time"] = x_df["Residence Time"].values

        dfs_mean.append(traj_mean_df)
        dfs_var.append(traj_var_df)

    # Concatenate all trajectories back together
    mean_df = pd.concat(dfs_mean, ignore_index=True)
    var_df = pd.concat(dfs_var, ignore_index=True)

    # Sort to match original order (optional)
    mean_df = mean_df.sort_values(
        ["SOLVENT NAME", "Temperature", "Residence Time"]
    ).reset_index(drop=True)
    var_df = var_df.sort_values(
        ["SOLVENT NAME", "Temperature", "Residence Time"]
    ).reset_index(drop=True)

    mean = mean_df[[col for col in mean_df.columns if col.startswith("state_")]].values
    var = var_df[[col for col in var_df.columns if col.startswith("state_")]].values

    return mean, var


def _normal_kl(mu1, var1, mu2, var2):
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
    kl = 0.5 * ((var1 / var2) + ((mu2 - mu1) ** 2) / var2 - 1 + torch.log(var2 / var1))
    return kl.sum(-1)


def find_latest_ckpt(train_dir, prefix="ckpt_epoch_", suffix=".pth"):
    import glob
    ckpt_files = glob.glob(os.path.join(train_dir, f"{prefix}*{suffix}"))
    if not ckpt_files:
        return None, None

    # Extract epoch numbers and corresponding file paths
    ckpts = [(int(f.split(prefix)[-1].split(suffix)[0]), f) for f in ckpt_files]
    ckpts.sort()  # sort by epoch number

    ckpt_epoch, ckpt_path = ckpts[-1]
    return ckpt_epoch, ckpt_path  # return the path of the largest epoch