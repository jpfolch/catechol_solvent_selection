from pathlib import Path

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
