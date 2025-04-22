import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT, TARGET_LABELS
from catechol.plots import style


def plot_solvent_prediction(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    assert set(INPUT_LABELS_SINGLE_SOLVENT).issubset(predictions.columns)
    fig, ax = plt.subplots()
    # sort by residence time
    sorted_idcs = np.argsort(predictions["Residence Time"])
    predictions = predictions.iloc[sorted_idcs]
    ground_truth = ground_truth.iloc[sorted_idcs]
    # filter by temperature
    high_temp_idcs = np.nonzero(predictions["Temperature"] == 225)
    predictions = predictions.iloc[high_temp_idcs]
    ground_truth = ground_truth.iloc[high_temp_idcs]
    for target in TARGET_LABELS:
        color = style.TARGET_TO_COLOR[target]
        ax.plot(
            predictions["Residence Time"],
            predictions[f"{target} mean"],
            color=color,
            label=target,
        )

        k = 1.0
        ax.fill_between(
            predictions["Residence Time"],
            predictions[f"{target} mean"] - k * np.sqrt(predictions[f"{target} var"]),
            predictions[f"{target} mean"] + k * np.sqrt(predictions[f"{target} var"]),
            color=color,
            alpha=0.2,
        )

        ax.scatter(
            predictions["Residence Time"],
            ground_truth[target],
            color=color,
            edgecolors="black",
            label="Ground Truth",
        )

    ax.legend()
    ax.set_xlabel("Residence Time (min)")
    ax.set_ylabel("Yield (%)")
    return ax
