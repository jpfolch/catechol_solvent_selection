def get_data_labels_mean_var() -> tuple[list[str], list[str]]:
    """Get the labels of the mean and variance predictions."""
    mean_labels = [f"{label} mean" for label in TARGET_LABELS]
    var_labels = [f"{label} var" for label in TARGET_LABELS]

    return mean_labels, var_labels


TARGET_LABELS = [
    "Product 1 yield",
    "Product 2 yield",
]