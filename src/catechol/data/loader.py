from pathlib import Path

import numpy as np
import pandas as pd

from catechol.data.data_labels import TARGET_LABELS


def load_single_solvent_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train X and Y dataframes for the single solvent experiments."""
    path = Path("data/single_solvent/catechol_single_solvent_yields.csv")
    assert path.exists(), f"Experiment data does not exist at {path.absolute()}"
    experiments = pd.read_csv(path)
    input_cols = [
        column for column in experiments.columns if column not in TARGET_LABELS
    ]
    return experiments[input_cols], experiments[TARGET_LABELS]


def train_test_split(
    df: pd.DataFrame, train_percentage: float, seed: int | None = None
):
    """Split a dataframe into a train/test pair.

    This is a naive way of splitting data. For evaluation with cross validation,
    and for methods where entire trajectories are observed (ODE), we should think
    about splitting based on experiment number."""

    num_train = int(len(df) * train_percentage)
    rng = np.random.default_rng(seed)
    train_idcs = rng.choice(len(df), size=num_train, replace=False)

    train_idcs_mask = np.zeros((len(df),), dtype=bool)
    train_idcs_mask[train_idcs] = True
    return df[train_idcs_mask], df[~train_idcs_mask]
