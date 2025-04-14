from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from catechol.data.data_labels import get_data_labels_mean_var
from .base_model import Model

class GPModel(Model):
    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        train_X_tensor = torch.tensor(train_X.to_numpy(), dtype=torch.float64)
        train_Y_tensor = torch.tensor(train_Y.to_numpy(), dtype=torch.float64)
        self.model = SingleTaskGP(
            train_X_tensor, train_Y_tensor, outcome_transform=None
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        test_X_tensor = torch.from_numpy(test_X.to_numpy()).to(torch.float64)
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
