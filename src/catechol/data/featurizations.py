from pathlib import Path
from typing import Literal

import pandas as pd

FEAT_DIRECTORY = Path("data/featurization_look_ups")
FEATURIZATIONS = ["acs_pca_descriptors", "drfps", "fragprints"]
FeaturizationType = Literal["acs_pca_descriptors", "drfps", "fragprints"]


def get_featurization_from_series(
    solvents: pd.Series, featurization: FeaturizationType
) -> pd.DataFrame:
    """Return the featurizations for a sequence of solvents.

    The output DataFrame has the same index as `solvents`, and a column containing
    the names of the solvents which has the same name as `solvents`."""
    if featurization not in FEATURIZATIONS:
        raise ValueError(
            f"Expected featurization in {FEATURIZATIONS}; got {featurization}."
        )

    file_path = FEAT_DIRECTORY / f"{featurization}_lookup.csv"
    assert (
        file_path.exists()
    ), f"Featurization lookup does not exist at {file_path.absolute()}"

    featurization_lookup = pd.read_csv(file_path, index_col=0).rename_axis(
        solvents.name, axis="index"
    )
    features = featurization_lookup.loc[solvents]
    return features.reset_index().set_index(solvents.index)


def featurize_input_df(
    X_df: pd.DataFrame, featurization: FeaturizationType
) -> pd.DataFrame:
    """Replace the SOLVENT NAME column(s) with their featurized representation."""
    for solvent_name_column in ["SOLVENT NAME", "SOLVENT A NAME", "SOLVENT B NAME"]:
        if solvent_name_column not in X_df.columns:
            continue

        feat = get_featurization_from_series(X_df[solvent_name_column], featurization)
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
