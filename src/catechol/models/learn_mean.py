import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood


class LearnMean(Mean):
    def __init__(
        self, X: pd.DataFrame, Y: pd.DataFrame, batch_shape=torch.Size(), **kwargs
    ):
        super(LearnMean, self).__init__()
        self.batch_shape = batch_shape

        # extract the residence time and temperature
        X = X[["Residence Time", "Temperature"]]
        # transform data into tensor
        X = torch.tensor(X.to_numpy(), dtype=torch.float64)
        Y = torch.tensor(Y.to_numpy(), dtype=torch.float64)

        self.prior_model = SingleTaskGP(X, Y, **kwargs)
        mll = ExactMarginalLogLikelihood(self.prior_model.likelihood, self.prior_model)
        fit_gpytorch_mll(mll, optimizer_kwargs=dict(timeout_sec=30))

    def forward(self, input):
        # extract the first two columns of the input
        X = input[..., :2]
        mean = self._predict_mean(X)
        # if input.shape[:-2] == self.batch_shape:
        #     return mean.expand(input.shape[:-1])
        # else:
        #     return mean.expand(_mul_broadcast_shape(input.shape[:-1], mean.shape))
        return mean[:, :, 0]

    def _predict_mean(self, test_X):
        posterior = self.prior_model.posterior(test_X)
        return posterior.mean
