import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from catechol.data.data_labels import get_data_labels_mean_var, is_df_solvent_ramp_dataset
from catechol.data.featurizations import FeaturizationType, featurize_input_df

from .base_model import Model


class GPModel(Model):
    def __init__(
        self, multitask: bool = False, featurization: FeaturizationType | None = None
    ):
        super().__init__(featurization=featurization)
        self.multitiask = multitask

    def _get_mixed_solvent_representation(self, X_featurized: pd.DataFrame):
        alpha = X_featurized["SolventB%"]
        def get_solvent_feat(solvent: str):
            feat = X_featurized.loc[:, X_featurized.columns.str.startswith(f"{solvent}_")]
            feat = feat.rename(columns=lambda c: c.removeprefix(f"{solvent}_"))
            return feat
        
        A_feat = get_solvent_feat("A")
        B_feat = get_solvent_feat("B")

        mixed_feat = A_feat.mul(1 - alpha, axis=0) + B_feat.mul(alpha, axis=0)
        return pd.concat((
            X_featurized[["Residence Time", "Temperature"]],
            mixed_feat,
        ), axis="columns")




    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        train_X_featurized = featurize_input_df(train_X, self.featurization, remove_constant=True, normalize_feats=True)
        if is_df_solvent_ramp_dataset(train_X):
            train_X_featurized = self._get_mixed_solvent_representation(train_X_featurized)

        train_X_tensor = torch.tensor(
            train_X_featurized.to_numpy(), dtype=torch.float64
        )
        train_Y_tensor = torch.tensor(train_Y.to_numpy(), dtype=torch.float64)

        model_cls = KroneckerMultiTaskGP if self.multitiask else SingleTaskGP
        self.model = model_cls(train_X_tensor, train_Y_tensor, mean_module=ZeroMean())

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        test_X_featurized = featurize_input_df(test_X, self.featurization, remove_constant=True, normalize_feats=True)
        if is_df_solvent_ramp_dataset(test_X):
            test_X_featurized = self._get_mixed_solvent_representation(test_X_featurized)
        
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
