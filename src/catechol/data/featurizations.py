from pathlib import Path
from typing import Literal

import pandas as pd

FEAT_DIRECTORY = Path("data/featurization_look_ups")
FEATURIZATIONS = [
    "acs_pca_descriptors",
    "drfps_catechol",
    "fragprints",
    "spange_descriptors",
    "smiles",
    "drfps_claisen",
]
FeaturizationType = Literal[
    "acs_pca_descriptors",
    "drfps_catechol",
    "fragprints",
    "spange_descriptors",
    "smiles",
    "drfps_claisen",
]


def _load_featurization_lookup(featurization: FeaturizationType):
    file_path = FEAT_DIRECTORY / f"{featurization}_lookup.csv"
    assert (
        file_path.exists()
    ), f"Featurization lookup does not exist at {file_path.absolute()}"
    return pd.read_csv(file_path, index_col=0)


def _remove_constant_values_from_featurization(
    featurization_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Remove any features that are constant across all solvents.
    This is useful for reducing the dimensionality of large featurizations."""
    constant_columns = (featurization_lookup == featurization_lookup.iloc[0]).all(
        axis=0
    )
    return featurization_lookup.loc[:, ~constant_columns]


def _normalize_featurization(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Map each column of the featurization to the interval [0, 1].

    This may not be suitable for PCA featurizations, where the relative lengthscale
    of each dimension may be important."""
    return (df - df.min()) / (df.max() - df.min())


def _get_featurization_from_series(
    solvents: pd.Series,
    featurization: FeaturizationType,
    remove_constant: bool,
    normalize_feats: bool,
) -> pd.DataFrame:
    """Return the featurizations for a sequence of solvents.

    The output DataFrame has the same index as `solvents`, and a column containing
    the names of the solvents which has the same name as `solvents`."""
    if featurization not in FEATURIZATIONS:
        raise ValueError(
            f"Expected featurization in {FEATURIZATIONS}; got {featurization}."
        )

    featurization_lookup = _load_featurization_lookup(featurization).rename_axis(
        solvents.name, axis="index"
    )

    if remove_constant:
        featurization_lookup = _remove_constant_values_from_featurization(
            featurization_lookup
        )

    if normalize_feats:
        featurization_lookup = _normalize_featurization(featurization_lookup)

    features = featurization_lookup.loc[solvents]
    return features.reset_index().set_index(solvents.index)


def featurize_input_df(
    X_df: pd.DataFrame,
    featurization: FeaturizationType,
    remove_constant: bool = False,
    normalize_feats: bool = False,
) -> pd.DataFrame:
    """Replace the SOLVENT NAME column(s) with their featurized representation.

    If `remove_constant`, then any features for which all solvents are equal are
    removed."""
    for solvent_name_column in ["SOLVENT NAME", "SOLVENT A NAME", "SOLVENT B NAME"]:
        if solvent_name_column not in X_df.columns:
            continue

        feat = _get_featurization_from_series(
            X_df[solvent_name_column], featurization, remove_constant, normalize_feats
        )
        if solvent_name_column in ["SOLVENT A NAME", "SOLVENT B NAME"]:
            # if this is the full dataset, extract either "A" or "B" from column name
            solvent_id = (
                solvent_name_column.replace("SOLVENT", "").replace("NAME", "").strip()
            )
            feat = feat.rename(
                columns=lambda col: col
                if col == solvent_name_column
                else f"{solvent_id}_{col}"
            )

        X_df = pd.concat(
            [
                X_df.drop(columns=solvent_name_column),
                feat.drop(columns=solvent_name_column),
            ],
            axis="columns",
        )

    return X_df
