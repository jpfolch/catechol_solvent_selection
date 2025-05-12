import numpy as np
import pandas as pd

from catechol.data.data_labels import get_data_labels_mean_var


def nlpd(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Calculate the negative log predictive density (NLPD) of the predictions."""
    mean_labels, var_labels = get_data_labels_mean_var(ground_truth.columns.to_list())
    mean = predictions[mean_labels].to_numpy()
    var = predictions[var_labels].to_numpy()

    y_true = ground_truth.to_numpy()
    nlpd = ((y_true - mean) ** 2) / (2 * var) + 0.5 * np.log(var)  # (N x M)
    return np.mean(np.sum(nlpd, axis=1))


def mse(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Calculate the mean squared error (MSE) of the predictions."""
    mean_labels, _ = get_data_labels_mean_var(ground_truth.columns.to_list())
    mean = predictions[mean_labels].to_numpy()
    y_true = ground_truth.to_numpy()
    mse = np.mean((y_true - mean) ** 2)
    return mse
