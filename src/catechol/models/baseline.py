import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from catechol.data.data_labels import (
    get_data_labels_mean_var,
)

from .base_model import Model


class BaselineModel(Model):
    """
    A naive model, that predicts using a Gaussian with the mean and
    variance of the training data.
    """

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame):
        self.means = train_Y.mean().to_numpy()
        self.stds = train_Y.std().to_numpy()

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        N = test_X.shape[0]
        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean_df = pd.DataFrame(
            np.tile(self.means, (N, 1)), index=test_X.index, columns=mean_lbl
        )
        var_df = pd.DataFrame(
            np.tile(self.stds, (N, 1)), index=test_X.index, columns=var_lbl
        )

        return pd.concat([mean_df, var_df], axis=1)

    def get_model_name(self):
        # overwrite the base Model because featurizations don't matter
        return self.__class__.__name__


class BaselineGPModel(Model):
    """
    A GP that fits only to temperature and lengthscale.
    """

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame):
        train_X_featurized = train_X[["Residence Time", "Temperature"]]
        train_X_tensor = torch.tensor(
            train_X_featurized.to_numpy(), dtype=torch.float64
        )
        train_Y_tensor = torch.tensor(train_Y.to_numpy(), dtype=torch.float64)

        model = SingleTaskGP(
            train_X_tensor,
            train_Y_tensor,
            mean_module=ZeroMean(),
        )

        self.model = model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, optimizer_kwargs=dict(timeout_sec=30))

        self.target_labels = train_Y.columns.to_list()

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        test_X_featurized = test_X[["Residence Time", "Temperature"]]
        test_X_tensor = torch.from_numpy(test_X_featurized.to_numpy()).to(torch.float64)
        with torch.no_grad():
            preds = self.model.posterior(test_X_tensor, observation_noise=True)
            mean = preds.mean.cpu().numpy()
            var = preds.variance.cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def get_model_name(self):
        # overwrite the base Model because featurizations don't matter
        return self.__class__.__name__
