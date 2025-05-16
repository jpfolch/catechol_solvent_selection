import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

from catechol import metrics
from catechol.data.data_labels import get_data_labels_mean_var
from catechol.data.featurizations import FeaturizationType, featurize_input_df
from catechol.data.loader import (
    replace_repeated_measurements_with_average,
    train_test_split,
)

from .base_model import Model

featurization_dims = {
    "acs_pca_descriptors": 5,
    "drfps": 2048,
    "fragprints": 2133,
    "spange_descriptors": 13,
}


class LODEModel(Model):
    """
    A Variational Inference based latent ODE
    Both the state and the dynamics are treated as latent variables
    """

    def __init__(
        self,
        device: str = "cpu",
        state_dim: int = 3,
        latent_state_dim: int = 32,
        latent_dynamics_dim: int = 32,
        h_dim_ode: int = 64,
        h_dim_x0: int = 64,
        h_dim_dynmcs: int = 64,
        h_dim_dec: int = 64,
        featurization: FeaturizationType | None = None,
        state_column_name: list[str] = ["Product 2", "Product 3", "SM"],
        learning_rate: float = 1e-3,
        train_epoch: int = 100,
    ):
        super().__init__(featurization=featurization)
        featurization_dim = featurization_dims[featurization]
        self.device = device
        self.latent_state_dim = latent_state_dim
        self.func = _ODEfunc_TIV_Dynmc(
            latent_state_dim, latent_dynamics_dim, h_dim_ode
        ).to(device)
        self.l_in_rec = _LantentInitCondRecogNet(
            latent_state_dim, h_dim_x0, featurization_dim + 1
        ).to(device)
        self.l_d_rec = _DynmcsRecogNet(
            latent_dynamics_dim, h_dim_dynmcs, featurization_dim + 1
        ).to(device)
        self.dec = _HeteroscedasticDecoder(
            latent_state_dim, h_dim_dec, state_dim, latent_dynamics_dim
        ).to(device)
        self.output_colnames = state_column_name
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch

    def _train(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        learning_rate: float = None,
        train_epoch: int = None,
        use_pretrained_model: bool = False,
        train_dir: str = None,
        kl_weight: float = 1.0,
        mc_sample_num: int = 1,
        validation_fraction: float = 0.0,
        **kwargs,
    ) -> None:
        if learning_rate is None:
            learning_rate = self.learning_rate
        if train_epoch is None:
            train_epoch = self.train_epoch

        # remove duplication, ensure time is strictly increasing
        train_X, train_Y = replace_repeated_measurements_with_average(train_X, train_Y)

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

        if validation_fraction != 0.0:
            train_X, val_X, train_Y, val_Y = train_test_split(
                train_X, train_Y, test_size=validation_fraction, random_state=1
            )

        train_X = _process_featurization(
            featurize_input_df(train_X, featurization=self.featurization),
            solvent_name=train_X["SOLVENT NAME"],
        )

        (
            keys,
            x_batch,
            y_batch,
            mask,
            meta,
            global_time_tensor,
        ) = _formulate_data_as_batch_tensor(train_X, train_Y)
        B, T, _ = x_batch.shape
        global_time = global_time_tensor.to(self.device)

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        mask = mask.to(self.device)
        temperature = meta["temperature"].to(self.device)
        solvent_embedding = meta["solvent_embedding"].to(self.device)

        latest_ckpt, latest_ckpt_path = _find_latest_ckpt(train_dir)
        if latest_ckpt_path is not None:
            checkpoint = torch.load(latest_ckpt_path)
            self.func.load_state_dict(checkpoint["func_state_dict"])
            self.l_in_rec.load_state_dict(checkpoint["l_in_rec_state_dict"])
            self.l_d_rec.load_state_dict(checkpoint["l_d_rec_state_dict"])
            self.dec.load_state_dict(checkpoint["dec_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            latest_ckpt = 0

        if use_pretrained_model:
            return

        pbar = tqdm(range(latest_ckpt, train_epoch + 1))
        for itr in pbar:
            optimizer.zero_grad()

            context = torch.cat([solvent_embedding, temperature], dim=-1)
            qz0_mean, qz0_std = self.l_in_rec(context)
            qzd_mean, qzd_std = self.l_d_rec(context)

            eps_z0 = torch.randn(mc_sample_num, *qz0_mean.shape).to(self.device)
            eps_zd = torch.randn(mc_sample_num, *qzd_mean.shape).to(self.device)

            z0 = (eps_z0 * qz0_std + qz0_mean).permute(1, 0, 2)
            zd = (eps_zd * qzd_std + qzd_mean).permute(1, 0, 2)

            pred_z = odeint(
                self.func,
                torch.cat([z0, zd], dim=-1),
                global_time,
            ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

            pred_mean, pred_std = self.dec(global_time, pred_z, z0, zd)
            pred_mean = pred_mean.permute(2, 0, 1, 3).permute(1, 2, 0, 3)
            pred_std = pred_std.permute(2, 0, 1, 3).permute(1, 2, 0, 3)

            logpx = (
                torch.distributions.Normal(pred_mean, pred_std).log_prob(
                    y_batch.unsqueeze(1)
                )
                * mask.unsqueeze(1).unsqueeze(-1)
            ).sum(-1).sum(-1) / mask.sum(-1)[..., None]
            logpx = logpx.mean(-1).mean(-1)

            kl_z0 = _normal_kl(
                qz0_mean,
                qz0_std**2,
                torch.zeros_like(qz0_mean),
                torch.ones_like(qz0_std),
            ).sum(-1)
            kl_zd = _normal_kl(
                qzd_mean,
                qzd_std**2,
                torch.zeros_like(qzd_mean),
                torch.ones_like(qzd_std),
            ).sum(-1)

            loss = torch.mean(-logpx + kl_weight * (kl_z0 + kl_zd))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=kwargs.get("grad_clip_max_norm", 2.0)
            )
            optimizer.step()

            mse = (
                (
                    (
                        (pred_mean - y_batch.unsqueeze(1)) ** 2
                        * mask.unsqueeze(1).unsqueeze(-1)
                    )
                    .sum(-1)
                    .sum(-1)
                    / mask.sum(-1)[..., None]
                )
                .mean(-1)
                .mean(-1)
            )

            if validation_fraction != 0.0:
                _predictions = self._predict(val_X, mc_sample_num=32)
                val_mse = metrics.mse(_predictions, val_Y)
            else:
                val_mse = np.inf

            pbar.set_postfix({"loss_mse": float(mse), "val_mse": val_mse})

            if itr % kwargs.get("save_freq", torch.inf) == 0 and train_dir is not None:
                os.makedirs(train_dir, exist_ok=True)
                ckpt_path = os.path.join(train_dir, f"ckpt_epoch_{itr}.pth")
                torch.save(
                    {
                        "func_state_dict": self.func.state_dict(),
                        "l_in_rec_state_dict": self.l_in_rec.state_dict(),
                        "l_d_rec_state_dict": self.l_d_rec.state_dict(),
                        "dec_state_dict": self.dec.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )

    def _predict(self, test_X: pd.DataFrame, mc_sample_num: int = 1) -> pd.DataFrame:
        self.func.eval()

        test_X_proc = _process_featurization(
            featurize_input_df(test_X, featurization=self.featurization),
            solvent_name=test_X["SOLVENT NAME"],
        )

        (
            keys,
            x_batch,
            _,
            mask,
            meta,
            global_time_tensor,
        ) = _formulate_data_as_batch_tensor(test_X_proc, y=None)
        B, _, _ = x_batch.shape
        mask = mask.to(self.device)
        global_time = global_time_tensor.to(self.device)
        temperature = meta["temperature"].to(self.device)
        solvent_embedding = meta["solvent_embedding"].to(self.device)

        context = torch.cat([solvent_embedding, temperature], dim=-1)
        with torch.no_grad():
            qz0_mean, qz0_std = self.l_in_rec(context)
            qzd_mean, qzd_std = self.l_d_rec(context)

            eps_z0 = torch.randn(mc_sample_num, *qz0_mean.shape).to(self.device)
            eps_zd = torch.randn(mc_sample_num, *qzd_mean.shape).to(self.device)

            z0 = (eps_z0 * qz0_std + qz0_mean).permute(1, 0, 2)
            zd = (eps_zd * qzd_std + qzd_mean).permute(1, 0, 2)

            pred_z = odeint(
                self.func,
                torch.cat([z0, zd], dim=-1),
                global_time,
            ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

            pred_mean, pred_std = self.dec(global_time, pred_z, z0, zd)
            pred_mean = pred_mean.permute(2, 0, 1, 3).permute(1, 2, 0, 3).mean(1)
            pred_std = pred_std.permute(2, 0, 1, 3).permute(1, 2, 0, 3).mean(1)

        pred_mean = pred_mean.cpu().numpy()
        pred_std = pred_std.cpu().numpy()
        mask_np = mask.cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var()

        full_mean, full_var = [], []
        for i, (key, mean_seq, std_seq, mask_seq) in enumerate(
            zip(keys, pred_mean, pred_std, mask_np)
        ):
            valid_rows = mask_seq.astype(bool)
            mean_df_i = pd.DataFrame(mean_seq[valid_rows], columns=mean_lbl)
            var_df_i = pd.DataFrame((std_seq[valid_rows] ** 2), columns=var_lbl)
            full_mean.append(mean_df_i)
            full_var.append(var_df_i)

        mean_df = pd.concat(full_mean, ignore_index=True)
        var_df = pd.concat(full_var, ignore_index=True)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass


class EODEModel(Model):
    """
    A Variational Inference based ODE
    The State is explicit (i.e., only consists of 'Product 2', 'Product 3', 'SM')
    The dynamics are treated as latent variables and inferred using VI
    """

    # normalize_inputs = False

    def __init__(
        self,
        device: str = "cpu",
        state_dim: int = 3,
        dynamics_dim: int = 32,
        h_dim_ode: int = 64,
        h_dim_dynmcs: int = 64,
        h_dim_dec: int = 64,
        featurization: FeaturizationType | None = None,
        state_column_name: list[str] = ["Product 2", "Product 3", "SM"],
        learning_rate: float = 1e-3,
        train_epoch: int = 100,
    ):
        super().__init__(featurization=featurization)
        featurization_dim = featurization_dims[featurization]
        self.device = device
        self.latent_state_dim = state_dim
        self.func = _ODEfunc_TIV_Dynmc(state_dim, dynamics_dim, h_dim_ode).to(device)
        self.l_d_rec = _DynmcsRecogNet(
            dynamics_dim, h_dim_dynmcs, featurization_dim + 1
        ).to(device)
        self.dec = _HeteroscedasticDecoder(
            state_dim, h_dim_dec, state_dim, dynamics_dim
        ).to(device)
        self.output_colnames = state_column_name
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch

    def _train(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        learning_rate: float = None,
        train_epoch: int = None,
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

        if learning_rate is None:
            learning_rate = self.learning_rate
        if train_epoch is None:
            train_epoch = self.train_epoch

        # remove duplication, ensure time is strictly increasing
        train_X, train_Y = replace_repeated_measurements_with_average(train_X, train_Y)

        optimizer = optim.Adam(params, lr=learning_rate)

        # data preprocessing: process the data to trajectories

        if validation_fraction != 0.0:  # validation fraction enabled
            train_X, val_X = train_test_split(
                train_X, train_percentage=1 - validation_fraction, seed=1
            )
            train_Y, val_Y = train_test_split(
                train_Y, train_percentage=1 - validation_fraction, seed=1
            )

        # data preprocessing: process the data to trajectories
        train_X = _process_featurization(
            featurize_input_df(train_X, featurization=self.featurization),
            solvent_name=train_X["SOLVENT NAME"],
        )

        (
            keys,
            x_batch,
            y_batch,
            mask,
            meta,
            global_time_tensor,
        ) = _formulate_data_as_batch_tensor(train_X, train_Y)
        B, _, _ = x_batch.shape
        global_time = global_time_tensor.to(self.device)

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        mask = mask.to(self.device)

        latest_ckpt, latest_ckpt_path = _find_latest_ckpt(train_dir)
        if latest_ckpt_path is not None:
            checkpoint = torch.load(latest_ckpt_path)
            self.func.load_state_dict(checkpoint["func_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded ckpt from {}".format(latest_ckpt_path))
        else:
            latest_ckpt = 0
            print("No existing ckpt found")

        if use_pretrained_model:
            return

        pbar = tqdm(range(latest_ckpt, train_epoch + 1))
        for itr in pbar:
            # TODO:
            optimizer.zero_grad()
            # infer and sample latent dynamics
            qzd_mean, qzd_std = self.l_d_rec.forward(
                torch.concat(
                    [meta["solvent_embedding"], meta["temperature"]], axis=-1
                ).to(self.device)
            )

            # Sample MC samples in a batch
            z0 = (
                torch.Tensor([[[0.0, 0.0, 1.0]]])
                .repeat(qzd_mean.shape[0], mc_sample_num, 1)
                .to(qzd_mean.device)
            )  # [batch_dim, mc_sample_num, latent_dim]

            epsilon_zd = torch.randn(mc_sample_num, *qzd_mean.size()).to(self.device)
            zd = (epsilon_zd * qzd_std + qzd_mean).permute(
                1, 0, 2
            )  # [batch_dim, mc_sample_num, latent_dim]

            # forward in time and solve ode for reconstructions
            pred_z = odeint(
                self.func,
                torch.concat([z0, zd], axis=-1),
                torch.Tensor(global_time).to(self.device),
            ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

            pred_mean, pred_std = self.dec(global_time, pred_z, z0, zd)
            pred_mean = pred_mean.permute(2, 0, 1, 3).permute(1, 2, 0, 3)
            pred_std = pred_std.permute(2, 0, 1, 3).permute(1, 2, 0, 3)

            # compute masked ELBO loss
            logpx = (
                (
                    (
                        torch.distributions.normal.Normal(
                            loc=pred_mean, scale=pred_std
                        ).log_prob(y_batch.unsqueeze(1))
                        * mask.unsqueeze(1).unsqueeze(-1)
                    )
                    .sum(-1)
                    .sum(-1)
                    / mask.sum(-1)[..., None]
                )
                .mean(-1)
                .mean(-1)
            )  # average across mc and batch

            pzd_mean = torch.zeros_like(qzd_mean).to(self.device)
            analytic_kl_qzd = _normal_kl(
                qzd_mean, qzd_std**2, pzd_mean, torch.ones_like(qzd_std)
            ).sum(-1)
            loss = torch.mean(-logpx + kl_weight * (analytic_kl_qzd), dim=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=kwargs.get("grad_clip_max_norm", 2.0)
            )
            optimizer.step()

            mse_x = (
                (
                    (
                        (
                            ((pred_mean - y_batch.unsqueeze(1)) ** 2)
                            * mask.unsqueeze(1).unsqueeze(-1)
                        )
                        .sum(-1)
                        .sum(-1)
                    )
                    / mask.sum(-1)[..., None]
                )
                .mean(-1)
                .mean(-1)
            )

            # pred val
            if validation_fraction != 0.0:
                _predictions = self._predict(val_X, mc_sample_num=32)
                val_mse = metrics.mse(_predictions, val_Y)
            else:
                val_mse = np.inf

            pbar.set_postfix(
                {
                    "avg_logpx": "{:2f}".format(float(logpx.detach())),
                    "avg_kl_zd": "{:2f}".format(float(analytic_kl_qzd.detach())),
                    "loss_mse": "{:2f}".format(float(mse_x.detach())),
                    "val_mse": "{:.2f}".format(val_mse),
                }
            )

        if itr % kwargs.get("save_freq", torch.inf) == 0 and train_dir is not None:
            os.makedirs(train_dir, exist_ok=True)
            ckpt_path = os.path.join(train_dir, f"ckpt_epoch_{itr}.pth")
            torch.save(
                {
                    "func_state_dict": self.func.state_dict(),
                    "l_d_rec_state_dict": self.l_d_rec.state_dict(),
                    "dec_state_dict": self.dec.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Stored ckpt at {ckpt_path}")

    def _predict(self, test_X: pd.DataFrame, mc_sample_num: int = 1) -> pd.DataFrame:
        self.func.eval()

        test_X_proc = _process_featurization(
            featurize_input_df(test_X, featurization=self.featurization),
            solvent_name=test_X["SOLVENT NAME"],
        )

        (
            keys,
            x_batch,
            _,
            mask,
            meta,
            global_time_tensor,
        ) = _formulate_data_as_batch_tensor(test_X_proc, y=None)
        B, _, _ = x_batch.shape
        x_batch = x_batch.to(self.device)
        mask = mask.to(self.device)
        global_time = global_time_tensor.to(self.device)
        temperature = meta["temperature"].to(self.device)
        solvent_embedding = meta["solvent_embedding"].to(self.device)

        z0 = (
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
            .repeat(B, mc_sample_num, 1)
            .to(self.device)
        )

        solvent_emb_temperature = torch.cat([solvent_embedding, temperature], dim=-1)
        with torch.no_grad():
            # # infer and sample latent dynamics
            qzd_mean, qzd_std = self.l_d_rec.forward(solvent_emb_temperature)

            # Sample MC samples in a batch
            epsilon_zd = torch.randn(mc_sample_num, *qzd_mean.size()).to(self.device)
            zd = (epsilon_zd * qzd_std + qzd_mean).permute(
                1, 0, 2
            )  # [batch_dim, mc_sample_num, latent_dim]

            # forward in time and solve ode for reconstructions
            pred_z = odeint(
                self.func,
                torch.concat([z0, zd], axis=-1),
                torch.Tensor(global_time).to(self.device),
            ).permute(1, 2, 0, 3)[..., : self.latent_state_dim]

            pred_mean, pred_std = self.dec(global_time, pred_z, z0, zd)
            pred_mean = (
                pred_mean.permute(2, 0, 1, 3).permute(1, 2, 0, 3).mean(1)
            )  # avg through MC dim
            pred_std = (
                pred_std.permute(2, 0, 1, 3).permute(1, 2, 0, 3).mean(1)
            )  # avg through MC dim

        pred_mean = pred_mean.cpu().numpy()  # [B, T, D]
        pred_std = pred_std.cpu().numpy()  # [B, T, D]
        mask_np = mask.cpu().numpy()  # [B, T]

        mean_lbl, var_lbl = get_data_labels_mean_var()

        full_mean, full_var = [], []
        for i, (key, mean_seq, std_seq, mask_seq) in enumerate(
            zip(keys, pred_mean, pred_std, mask_np)
        ):
            valid_rows = mask_seq.astype(bool)
            mean_df_i = pd.DataFrame(mean_seq[valid_rows], columns=mean_lbl)
            var_df_i = pd.DataFrame((std_seq[valid_rows] ** 2), columns=var_lbl)
            full_mean.append(mean_df_i)
            full_var.append(var_df_i)

        mean_df = pd.concat(full_mean, ignore_index=True)
        var_df = pd.concat(full_var, ignore_index=True)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass


class NODEModel(Model):
    """
    An explicit, None Neural ODE model with a temperal dynamics hypernetwork
    The model has a hypernetwork to infer time-dependent dynamics
    characterize likelihood p(yt|xt) enabling MLE training
    """

    # normalize_inputs = True

    def __init__(
        self,
        device: str = "cpu",
        state_dim: int = 3,
        h_dim_ode: int = 64,
        featurization: FeaturizationType | None = None,
        state_column_name: list[str] = ["Product 2", "Product 3", "SM"],
        learning_rate: float = 1e-3,
        train_epoch: int = 100,
    ):
        super().__init__(featurization=featurization)
        featurization_dim = featurization_dims[featurization]
        self.device = device
        self.state_dim = state_dim

        self.func = _ODEfunc_TV_Dynmv(state_dim, featurization_dim + 1, h_dim_ode).to(
            device
        )
        self.output_colnames = state_column_name
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch

    def _train(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        learning_rate: float = None,
        train_epoch: int = None,
        use_pretrained_model: bool = False,
        train_dir: str = None,
        validation_fraction: float = 0.0,
        **kwargs,
    ) -> None:
        if learning_rate is None:
            learning_rate = self.learning_rate
        if train_epoch is None:
            train_epoch = self.train_epoch

        # remove duplication, ensure time is strictly increasing
        train_X, train_Y = replace_repeated_measurements_with_average(train_X, train_Y)

        # initialize
        self.func.to(self.device)

        params = list(self.func.parameters())
        optimizer = optim.Adam(params, lr=learning_rate)

        if validation_fraction != 0.0:  # validation fraction enabled
            train_X, val_X = train_test_split(
                train_X, train_percentage=1 - validation_fraction, seed=1
            )
            train_Y, val_Y = train_test_split(
                train_Y, train_percentage=1 - validation_fraction, seed=1
            )

        # data preprocessing: process the data to trajectories
        train_X = _process_featurization(
            featurize_input_df(train_X, featurization=self.featurization),
            solvent_name=train_X["SOLVENT NAME"],
        )

        (
            keys,
            x_batch,
            y_batch,
            mask,
            meta,
            global_time_tensor,
        ) = _formulate_data_as_batch_tensor(train_X, train_Y)
        B, _, _ = x_batch.shape
        global_time = global_time_tensor.to(self.device)

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        mask = mask.to(self.device)
        temperature = meta["temperature"].to(self.device)  # [B, 1]
        solvent_embedding = meta["solvent_embedding"].to(self.device)  # [B, D_embed]

        latest_ckpt, latest_ckpt_path = _find_latest_ckpt(train_dir)
        if latest_ckpt_path is not None:
            checkpoint = torch.load(latest_ckpt_path)
            self.func.load_state_dict(checkpoint["func_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded ckpt from {}".format(latest_ckpt_path))
        else:
            latest_ckpt = 0
            print("No existing ckpt found")

        if use_pretrained_model:
            return

        pbar = tqdm(range(latest_ckpt, train_epoch + 1))
        for itr in pbar:
            optimizer.zero_grad()

            # initial state z0 ∈ [B, latent_dim]
            z0 = (
                torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
                .repeat(B, 1)
                .to(self.device)
            )  # shape: [B, 3]

            # solve ODE
            def ode_rhs(t, x):  # x: [B, latent_dim]
                return self.func(t, x, solvent_embedding, temperature)

            z_pred = odeint(ode_rhs, z0, global_time)  # [T, B, latent_dim]

            z_pred = z_pred.permute(1, 0, 2)  # [B, T, latent_dim]

            # MSE loss with mask
            squared_error = (z_pred - y_batch) ** 2  # [B, T, D]
            masked_mse = (
                (squared_error * mask.unsqueeze(-1)).sum(-1).sum(-1) / mask.sum(-1)
            ).mean()

            loss = masked_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=kwargs.get("grad_clip_max_norm", 2.0)
            )
            optimizer.step()

            if validation_fraction != 0.0:
                _predictions = self._predict(val_X)
                val_mse = metrics.mse(_predictions, val_Y)
            else:
                val_mse = np.inf

            pbar.set_postfix(
                {"loss_mse": loss.item(), "val_mse": "{:.4f}".format(val_mse)}
            )

            if itr % kwargs.get("save_freq", torch.inf) == 0 and train_dir is not None:
                os.makedirs(train_dir, exist_ok=True)
                ckpt_path = os.path.join(train_dir, f"ckpt_epoch_{itr}.pth")
                torch.save(
                    {
                        "func_state_dict": self.func.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"Stored ckpt at {ckpt_path}")

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the trained ODE model.
        Returns:
            DataFrame with predicted outputs, using self.output_colnames as column names.
            Keeps strict row order alignment with test_X.
        """
        self.func.eval()

        test_X_proc = _process_featurization(
            featurize_input_df(test_X, featurization=self.featurization),
            solvent_name=test_X["SOLVENT NAME"],
        )

        (
            keys,
            x_batch,
            _,
            mask,
            meta,
            global_time_tensor,
        ) = _formulate_data_as_batch_tensor(test_X_proc, y=None)
        B, _, _ = x_batch.shape
        x_batch = x_batch.to(self.device)
        mask = mask.to(self.device)
        global_time = global_time_tensor.to(self.device)
        temperature = meta["temperature"].to(self.device)
        solvent_embedding = meta["solvent_embedding"].to(self.device)

        z0 = (
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
            .repeat(B, 1)
            .to(self.device)
        )

        with torch.no_grad():

            def ode_rhs(t, x):
                return self.func(t, x, solvent_embedding, temperature)

            z_pred = odeint(ode_rhs, z0, global_time)  # [T, B, D]
            z_pred = z_pred.permute(1, 0, 2)  # [B, T, D]

        z_pred_np = z_pred.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Reconstruct DataFrame per batch index in original order
        full_outputs = []
        for i, (key, z_seq, m_seq) in enumerate(zip(keys, z_pred_np, mask_np)):
            valid_rows = m_seq.astype(bool)
            pred_df = pd.DataFrame(z_seq[valid_rows], columns=self.output_colnames)
            full_outputs.append(pred_df)

        final = pd.concat(full_outputs, ignore_index=True)
        mean_lbl, var_lbl = get_data_labels_mean_var()
        final.columns = mean_lbl
        mean_df = final
        var_df = pd.DataFrame(np.zeros_like(mean_df), columns=var_lbl)

        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass


class _ODEfunc_TIV_Dynmc(nn.Module):
    def __init__(self, latent_dim=4, latent_dynamics_dim=4, nhidden=20):
        self.latent_dynamics_dim = latent_dynamics_dim
        super(_ODEfunc_TIV_Dynmc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim + latent_dynamics_dim + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        zd = x[..., -self.latent_dynamics_dim :]
        # augment t as input
        expand_t = torch.ones(zd.shape[:-1]).unsqueeze(-1).to(x.device) * t
        out = self.fc1(torch.concatenate([expand_t, x], axis=-1))
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return torch.concatenate([out, zd], axis=-1)


class _ODEfunc_TV_Dynmv(nn.Module):
    def __init__(self, latent_dim, dynamics_characterization_dim, nhidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.elu = nn.ELU(inplace=True)

        # Hypernetwork that maps solvent+temperature to a context vector zd
        self.dynamics_hypernetwork = nn.Sequential(
            nn.Linear(dynamics_characterization_dim + 1, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
        )

        # ODE dynamics conditioned on zd
        self.fc1 = nn.Linear(latent_dim + nhidden + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)

        self.nfe = 0  # number of function evaluations

    def forward(self, t, x, solvent_feature, temperature):
        """
        Args:
            t: scalar float (Python float or 0-D torch tensor)
            x: [B, latent_dim]
            solvent_feature: [B, D_s]
            temperature: [B] or [B, 1]
        Returns:
            dx/dt: [B, latent_dim]
        """
        self.nfe += 1
        if temperature.ndim == 1:
            temperature = temperature.unsqueeze(-1)  # [B, 1]

        t_expand = torch.ones_like(temperature) * t  # [B, 1]
        hyper_input = torch.cat(
            [solvent_feature, temperature, t_expand], dim=-1
        )  # [B, D_s + 2]

        zd = self.dynamics_hypernetwork(hyper_input)  # [B, D_dyn]

        # t_input = torch.ones_like(x[..., :1]).to(x.device) * t  # [B, 1]
        h = torch.cat([x, t_expand, zd], dim=-1)  # [B, latent_dim + 1 + D_dyn]

        out = self.elu(self.fc1(h))
        out = self.elu(self.fc2(out))
        dxdt = self.fc3(out)
        return dxdt


class _LantentInitCondRecogNet(nn.Module):
    def __init__(self, latent_init_dim, latent_dim, obs_dim, std_lower_bound=1e-2):
        super(_LantentInitCondRecogNet, self).__init__()
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


class _DynmcsRecogNet(nn.Module):
    def __init__(self, latent_dynamics_dim, latent_dim, obs_dim, std_lower_bound=1e-2):
        super(_DynmcsRecogNet, self).__init__()
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
    def __init__(
        self, latent_state_dim, nhidden, obs_dim, dynamics_dim, std_lower_bound=1e-2
    ):
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


def _formulate_data_as_batch_tensor(
    x: pd.DataFrame, y: pd.DataFrame | None = None
) -> tuple[
    list[tuple[str, float]],
    torch.Tensor,  # x_batch: [B, T, Dx]
    torch.Tensor | None,  # y_batch: [B, T, Dy] or None
    torch.Tensor,  # mask:    [B, T]
    dict[
        str, torch.Tensor
    ],  # meta:  { "temperature": [B, 1], "solvent_embedding": [B, D_embed] }
    torch.Tensor,  # global_time_tensor: [T]
]:
    combined = pd.concat([x, y], axis=1) if y is not None else x.copy()
    grouped = list(combined.groupby(["Solvent Name", "Temperature"]))

    global_time = sorted(
        set(t for _, group in grouped for t in group["Residence Time"])
    )
    time_to_idx = {t: i for i, t in enumerate(global_time)}
    max_len = len(global_time)

    Dy = len(y.columns) if y is not None else 0
    Dx = len(combined["Features"].iloc[0])
    B = len(grouped)

    x_batch = np.zeros((B, max_len, Dx), dtype=np.float32)
    y_batch = np.zeros((B, max_len, Dy), dtype=np.float32) if y is not None else None
    mask = np.zeros((B, max_len), dtype=np.float32)

    temperature_list = []
    embedding_list = []
    keys = []

    for i, ((solvent, temp), group) in enumerate(grouped):
        keys.append((solvent, float(temp)))
        group = group.sort_values(by="Residence Time").reset_index(drop=True)
        temperature_list.append([float(temp)])
        embedding_list.append(
            group["Features"].iloc[0]
        )  # same embedding for all timesteps in this group

        for _, row in group.iterrows():
            t_idx = time_to_idx[row["Residence Time"]]
            x_batch[i, t_idx] = row["Features"]
            mask[i, t_idx] = 1.0
            if y is not None:
                y_batch[i, t_idx] = row[y.columns].to_numpy()

    x_batch = torch.tensor(x_batch, dtype=torch.float32)
    y_batch = torch.tensor(y_batch, dtype=torch.float32) if y is not None else None
    mask = torch.tensor(mask, dtype=torch.float32)
    temperature_tensor = torch.tensor(temperature_list, dtype=torch.float32)
    embedding_tensor = torch.tensor(embedding_list, dtype=torch.float32)

    meta = {"temperature": temperature_tensor, "solvent_embedding": embedding_tensor}

    global_time_tensor = torch.tensor(global_time, dtype=torch.float32)
    return keys, x_batch, y_batch, mask, meta, global_time_tensor


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


def _process_featurization(
    dataframe: pd.DataFrame, solvent_name: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Multiple Columns of featurization to single column
    """
    # Assume dataframe is the one shown in the image
    res_time = dataframe["Residence Time"]
    temperature = dataframe["Temperature"]
    features = dataframe.drop(columns=["Residence Time", "Temperature"])

    # Combine into new dataframe
    new_dataframe = pd.DataFrame(
        {
            "Solvent Name": solvent_name,
            "Residence Time": res_time,
            "Temperature": temperature,
            "Features": features.values.tolist(),  # Each row becomes a list of 2048 features
        }
    )
    return new_dataframe


def _find_latest_ckpt(train_dir, prefix="ckpt_epoch_", suffix=".pth"):
    try:
        ckpt_files = glob.glob(os.path.join(train_dir, f"{prefix}*{suffix}"))
    except:
        return None, None
    if not ckpt_files:
        return None, None

    # Extract epoch numbers and corresponding file paths
    ckpts = [(int(f.split(prefix)[-1].split(suffix)[0]), f) for f in ckpt_files]
    ckpts.sort()  # sort by epoch number

    ckpt_epoch, ckpt_path = ckpts[-1]
    return ckpt_epoch, ckpt_path  # return the path of the largest epoch
