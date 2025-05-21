import pandas as pd


def get_data_labels_mean_var(
    target_labels: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Get the labels of the mean and variance predictions."""
    if target_labels is None:
        target_labels = TARGET_LABELS
    mean_labels = [f"{label} mean" for label in target_labels]
    var_labels = [f"{label} var" for label in target_labels]

    return mean_labels, var_labels


def is_df_solvent_ramp_dataset(X: pd.DataFrame) -> bool:
    col_set = set(X.columns)
    full_data_set = set(INPUT_LABELS_FULL_DATA)
    single_solvent_set = set(INPUT_LABELS_SINGLE_SOLVENT)
    if full_data_set.issubset(col_set):
        return True
    if single_solvent_set.issubset(col_set):
        return False
    raise ValueError(
        f"Some columns are missing from training DataFrame {X.columns}. "
        "Could not identify whether the task is single solvent or solvent ramps."
    )


INPUT_LABELS_SINGLE_SOLVENT = [
    "Residence Time",
    "Temperature",
    "SOLVENT NAME",
]

INPUT_LABELS_FULL_DATA = [
    "Residence Time",
    "Temperature",
    "SOLVENT A NAME",
    "SOLVENT B NAME",
    "SolventB%",
]

INPUT_LABELS_ACTIVE_LEARNING = [
    "Residence Time",
    "Temperature",
    "SOLVENT A NAME",
    "SOLVENT B NAME",
    "SolventB%",
    "RAMP NUM",
]

TARGET_LABELS = [
    "Product 2",
    "Product 3",
    "SM",
]

TARGET_CLAISEN_LABELS = ["Product"]
