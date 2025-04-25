import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catechol.models import Model
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT, TARGET_LABELS
from catechol.plots import style

def _plot_model_mean_and_confidence(model: Model, solvent_name: str, temperature: float, ax: plt.Axes):
    test_X = pd.DataFrame(np.linspace(0.0, 15.0, 100), columns=["Residence Time"])
    test_X["Temperature"] = temperature
    test_X["SOLVENT NAME"] = solvent_name
    predictions = model.predict(test_X)

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

def _plot_ground_truth(test_X: pd.DataFrame, test_Y: pd.DataFrame, ax: plt.Axes):
    for target in TARGET_LABELS:
        color = style.TARGET_TO_COLOR[target]
        ax.scatter(
            test_X["Residence Time"],
            test_Y[target],
            color=color,
            edgecolors="black",
            label="Ground Truth",
        )

def plot_solvent_prediction(model: Model, test_X: pd.DataFrame, test_Y: pd.DataFrame) -> plt.Axes:
    fig, ax = plt.subplots()
    solvent = test_X["SOLVENT NAME"].unique()[0]
    _plot_model_mean_and_confidence(model, solvent, 225, ax)
    _plot_ground_truth(test_X, test_Y, ax)
    ax.legend()
    ax.set_xlabel("Residence Time (min)")
    ax.set_ylabel("Yield (%)")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f"Solvent: {solvent}")
    return ax
