import pandas as pd

BOUNDS = {
    "Residence Time": [0.0, 15.0],
    "Temperature": [175.0, 225.0],
}


def normalize(X: pd.DataFrame) -> pd.DataFrame:
    """Map values of X to the interval [0, 1] where bounds are known."""
    X_norm = X.copy()
    for column in X_norm.columns:
        if column in BOUNDS:
            c_min, c_max = BOUNDS[column]
            normalized = (X[column] - c_min) / (c_max - c_min)
            X_norm[column] = normalized

    return X_norm
