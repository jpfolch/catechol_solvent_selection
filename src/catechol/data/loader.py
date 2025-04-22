import warnings
from pathlib import Path
from typing import Any, Generator

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
    and for methods where entire trajectories are observed (ODE), we should
    split based on experiment number."""

    warnings.warn(
        "Since this data is a time series, this train/test split "
        "leads to easy interpolation, and is not a good reflection of "
        "model performance. Please use a different split method."
    )

    num_train = int(len(df) * train_percentage)
    rng = np.random.default_rng(seed)
    train_idcs = rng.choice(len(df), size=num_train, replace=False)

    train_idcs_mask = np.zeros((len(df),), dtype=bool)
    train_idcs_mask[train_idcs] = True
    return df[train_idcs_mask], df[~train_idcs_mask]


def generate_leave_one_out_splits(
    X: pd.DataFrame, Y: pd.DataFrame
) -> Generator[
    tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    Any,
    None,
]:
    """Generate all leave-one-out splits across the solvents.

    For each split, one of the solvents will be removed from the training set to
    make a test set.
    """

    all_solvents = X["SOLVENT NAME"].unique()
    for solvent_name in sorted(all_solvents):
        train_idcs_mask = X["SOLVENT NAME"] != solvent_name
        yield (
            (X[train_idcs_mask], Y[train_idcs_mask]),
            (X[~train_idcs_mask], Y[~train_idcs_mask]),
        )
