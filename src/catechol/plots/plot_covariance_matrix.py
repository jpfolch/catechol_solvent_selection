import matplotlib.pyplot as plt
import pandas as pd
import torch

from catechol.data.featurizations import featurize_input_df
from catechol.models import GPModel


def plot_gp_covariance_matrix(model: GPModel, train_X: pd.DataFrame):
    fig, ax = plt.subplots()

    train_X_solvents = pd.DataFrame(
        train_X["SOLVENT NAME"].unique(), columns=["SOLVENT NAME"]
    )
    train_X_solvents[["Residence Time", "Temperature"]] = [0.0, 0.0]

    train_X_featurized = featurize_input_df(
        train_X_solvents, model.featurization, remove_constant=True
    )
    train_X_tensor = torch.tensor(train_X_featurized.to_numpy(), dtype=torch.float64)

    with torch.no_grad():
        cov = model.model.covar_module(train_X_tensor)

    ax.imshow(cov[0].evaluate())
