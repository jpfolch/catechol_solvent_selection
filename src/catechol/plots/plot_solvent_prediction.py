import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catechol.models import Model
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT, TARGET_LABELS
from catechol.plots import style

def _plot_model_mean_and_confidence(model: Model, solvent_name: str, reaction_smiles: str, temperature: float, ax: plt.Axes):
    test_X = pd.DataFrame(np.linspace(0.0, 15.0, 100), columns=["Residence Time"])
    test_X["Temperature"] = temperature
    test_X["SOLVENT NAME"] = solvent_name
    if reaction_smiles is not None:
        test_X["Reaction SMILES"] = reaction_smiles
    predictions = model.predict(test_X)

    for target in TARGET_LABELS:
        color = style.TARGET_TO_COLOR[target]
        ax.plot(
            predictions["Residence Time"],
            predictions[f"{target} mean"],
            color=color,
            label=target,
        )

        k = 1.96
        ax.fill_between(
            predictions["Residence Time"],
            predictions[f"{target} mean"] - k * np.sqrt(predictions[f"{target} var"]),
            predictions[f"{target} mean"] + k * np.sqrt(predictions[f"{target} var"]),
            color=color,
            alpha=0.2,
        )

def _plot_ground_truth(test_X: pd.DataFrame, test_Y: pd.DataFrame, temperature: float, ax: plt.Axes):
    temp_mask = test_X["Temperature"] == temperature
    for target in TARGET_LABELS:
        color = style.TARGET_TO_COLOR[target]
        ax.scatter(
            test_X.loc[temp_mask, "Residence Time"],
            test_Y.loc[temp_mask, target],
            color=color,
            edgecolors="black",
            label="Ground Truth",
        )

def plot_solvent_prediction(model: Model, test_X: pd.DataFrame, test_Y: pd.DataFrame) -> plt.Axes:
    fig, axs = plt.subplots(ncols=2, figsize=(6, 4), sharey=True)
    solvent = test_X["SOLVENT NAME"].iloc[0]
    if "Reaction SMILES" in test_X.columns:
        reaction_smiles = test_X["Reaction SMILES"].unique()[0]
    else:
        reaction_smiles = None
    _plot_model_mean_and_confidence(model, solvent, reaction_smiles, 175, axs[0])
    _plot_model_mean_and_confidence(model, solvent, reaction_smiles, 225, axs[1])
    _plot_ground_truth(test_X, test_Y, 175, axs[0])
    _plot_ground_truth(test_X, test_Y, 225, axs[1])
    for ax in axs:
        ax.legend()
        ax.set_xlabel("Residence Time (min)")
        ax.set_ylim(-0.05, 1.0)

    axs[0].set_title("Temperature: 175C")
    axs[0].set_ylabel("Yield (%)")
    axs[1].set_title("Temperature: 225C")
    fig.suptitle(f"Solvent: {solvent}")
    return fig
