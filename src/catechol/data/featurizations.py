import pandas as pd
from typing import Literal
import numpy as np
from pathlib import Path

FEAT_DIRECTORY = Path("data/featurization_look_ups")
FEATURIZATIONS = ["acs_pca_descriptors", "drfps", "fragprints"]

def get_featurization(solvents: pd.Series, featurization: Literal["acs_pca_descriptors", "drfps", "fragprints"]) -> pd.DataFrame:
    """Return the featurizations for a sequence of solvents.
    
    The output DataFrame has the same index as `solvents`, and a column containing
    the names of the solvents which has the same name as `solvents`."""
    if featurization not in FEATURIZATIONS:
        raise ValueError(f"Expected featurization in {FEATURIZATIONS}; got {featurization}.")
    
    file_path = FEAT_DIRECTORY / f"{featurization}_lookup.csv"
    assert file_path.exists(), f"Featurization lookup does not exist at {file_path.absolute()}"

    featurization_lookup = pd.read_csv(file_path, index_col=0).rename_axis(solvents.name, axis="index")
    features = featurization_lookup.loc[solvents]
    return features.reset_index()