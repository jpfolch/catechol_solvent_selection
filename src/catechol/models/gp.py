import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from catechol.data.data_labels import (
    get_data_labels_mean_var,
    is_df_solvent_ramp_dataset,
)
from catechol.data.featurizations import FeaturizationType, featurize_input_df

from .base_model import Model


class GPModel(Model):
    def __init__(
        self,
        multitask: bool = False,
        use_input_warp: bool = False,
        featurization: FeaturizationType | None = None,
    ):
        super().__init__(featurization=featurization)
        self.multitask = multitask
        self.use_input_warp = use_input_warp

    def _get_mixed_solvent_representation(self, X_featurized: pd.DataFrame):
        alpha = X_featurized["SolventB%"]

        def get_solvent_feat(solvent: str):
            feat = X_featurized.loc[
                :, X_featurized.columns.str.startswith(f"{solvent}_")
            ]
            feat = feat.rename(columns=lambda c: c.removeprefix(f"{solvent}_"))
            return feat

        A_feat = get_solvent_feat("A")
        B_feat = get_solvent_feat("B")

        mixed_feat = A_feat.mul(1 - alpha, axis=0) + B_feat.mul(alpha, axis=0)
        return pd.concat(
            (
                X_featurized[["Residence Time", "Temperature"]],
                mixed_feat,
            ),
            axis="columns",
        )

    def _get_input_transform(self, train_X_featurized: pd.DataFrame):
        """Get the warping input transform."""
        if not self.use_input_warp:
            return None

        # We only want to warp the time and solvent mixture ratio
        warp_col_mask = train_X_featurized.columns.isin(("Residence Time", "SolventB%"))
        indices = np.argwhere(warp_col_mask).flatten().tolist()
        d = train_X_featurized.shape[-1]
        bounds = torch.tensor([[0.0] * d, [1.0] * d])

        return Warp(
            d,
            indices,
            concentration0_prior=LogNormalPrior(0.0, 0.30**0.5),
            concentration1_prior=LogNormalPrior(0.0, 0.30**0.5),
            bounds=bounds,
        )

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        train_X_featurized = featurize_input_df(
            train_X, self.featurization, remove_constant=True, normalize_feats=True
        )
        if is_df_solvent_ramp_dataset(train_X):
            train_X_featurized = self._get_mixed_solvent_representation(
                train_X_featurized
            )

        train_X_tensor = torch.tensor(
            train_X_featurized.to_numpy(), dtype=torch.float64
        )
        train_Y_tensor = torch.tensor(train_Y.to_numpy(), dtype=torch.float64)

        model_cls = KroneckerMultiTaskGP if self.multitask else SingleTaskGP
        warp = self._get_input_transform(train_X_featurized)

        self.model = model_cls(
            train_X_tensor, train_Y_tensor, mean_module=ZeroMean(), input_transform=warp
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, optimizer_kwargs=dict(timeout_sec=30))

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        test_X_featurized = featurize_input_df(
            test_X, self.featurization, remove_constant=True, normalize_feats=True
        )
        if is_df_solvent_ramp_dataset(test_X):
            test_X_featurized = self._get_mixed_solvent_representation(
                test_X_featurized
            )

        test_X_tensor = torch.from_numpy(test_X_featurized.to_numpy()).to(torch.float64)
        with torch.no_grad():
            preds = self.model.posterior(test_X_tensor)
            mean = preds.mean.cpu().numpy()
            var = preds.variance.cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var()
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass

    def _get_model_name(self) -> str:
        multi = "-multi" if self.multitask else "-indep"
        warp = "-warp" if self.use_input_warp else ""
        return f"{self.__class__.__name__}{multi}{warp}"
