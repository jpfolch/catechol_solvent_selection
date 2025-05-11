import numpy as np
import pandas as pd

from catechol.data.data_labels import get_data_labels_mean_var

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
        mean_lbl, var_lbl = get_data_labels_mean_var()
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
