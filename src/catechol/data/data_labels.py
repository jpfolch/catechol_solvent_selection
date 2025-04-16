def get_data_labels_mean_var() -> tuple[list[str], list[str]]:
    """Get the labels of the mean and variance predictions."""
    mean_labels = [f"{label} mean" for label in TARGET_LABELS]
    var_labels = [f"{label} var" for label in TARGET_LABELS]

    return mean_labels, var_labels

INPUT_LABELS_SINGLE_SOLVENT = [
    "Residence Time",
    "Temperature",
    "SOLVENT_NAME"
]

TARGET_LABELS = [
    "Product 1",
    "Product 2",
]
