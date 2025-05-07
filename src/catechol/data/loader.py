import warnings
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from catechol.data.data_labels import TARGET_LABELS


def replace_repeated_measurements_with_average(
    X: pd.DataFrame, Y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace any measurements taken at the same time/conditions with their mean.

    This can be used when training models, to avoid biasing towards this data, or
    during testing so that model performance is not overly weighted by prediction
    at datapoints with more observations.

    This will reduce the number of observations."""

    df = pd.concat((X, Y), axis="columns")
    # round residence time, as the time varies by +-0.05
    df["Residence Time"] = df["Residence Time"].round(decimals=1)
    grpd = df.groupby(by=X.columns.to_list()).mean().reset_index()
    X_avgd, Y_avgd = grpd[X.columns], grpd[Y.columns]
    return X_avgd, Y_avgd


def load_single_solvent_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train X and Y dataframes for the single solvent experiments."""
    path = Path("data/single_solvent/catechol_single_solvent_yields.csv")
    assert path.exists(), f"Experiment data does not exist at {path.absolute()}"
    experiments = pd.read_csv(path)
    input_cols = [
        column for column in experiments.columns if column not in TARGET_LABELS
    ]
    return experiments[input_cols], experiments[TARGET_LABELS]

def load_solvent_ramp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train X and Y dataframes for the solvent ramp experiments."""
    path = Path("data/full_data/catechol_full_data_yields.csv")
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

def generate_leave_one_ramp_out_splits(
          X: pd.DataFrame, Y: pd.DataFrame
) -> Generator[
    tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    Any,
    None,
]:
    """Generate all leave-one-out splits across the solvent ramps.
    
    For each split, one of the solvent ramps will be removed from the training
    set to make a test set.
    """

    all_solvent_ramps = X[["SOLVENT A NAME", "SOLVENT B NAME"]].drop_duplicates()
    all_solvent_ramps.sort_values(by=["SOLVENT A NAME", "SOLVENT B NAME"])
    for _, solvent_pair in all_solvent_ramps.iterrows():
        train_idcs_mask = (X[["SOLVENT A NAME", "SOLVENT B NAME"]] != solvent_pair).all(axis=1)
        yield (
            (X[train_idcs_mask], Y[train_idcs_mask]),
            (X[~train_idcs_mask], Y[~train_idcs_mask]),
        )