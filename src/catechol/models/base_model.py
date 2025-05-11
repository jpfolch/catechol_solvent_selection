from abc import ABC, abstractmethod

import pandas as pd

from catechol.data.featurizations import FeaturizationType
from catechol.data.normalize import normalize


class Model(ABC):
    """Base class for all models."""

    normalize_inputs = True
    extra_input_columns = []

    def __init__(self, featurization: FeaturizationType | None = None):
        """Initialize the model."""
        self.is_fitted = False
        self.featurization = featurization

    def train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame, *args, **kwargs) -> None:
        """Train the model on the given data."""
        if self.normalize_inputs:
            train_X = normalize(train_X)
        self._train(train_X, train_Y, *args, **kwargs)
        self.is_fitted = True

    @abstractmethod
    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        """Abstract method to train the model. Must be implemented by subclasses."""
        pass

    def predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        test_X_norm = normalize(test_X) if self.normalize_inputs else test_X

        pred = self._predict(test_X_norm).set_index(test_X_norm.index)
        return pd.concat([test_X, pred], axis=1)

    @abstractmethod
    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to make predictions. Must be implemented by subclasses."""
        pass

    def ask(self) -> pd.DataFrame:
        """Ask the model for a candidate experiment, for Bayesian optimization."""
        return self._ask()

    @abstractmethod
    def _ask(self) -> pd.DataFrame:
        """Abstract method to ask the model for a candidate experiment. Must be implemented by subclasses."""
        pass

    def get_model_name(self) -> str:
        return f"{self._get_model_name()}-{self.featurization}"

    def _get_model_name(self) -> str:
        return self.__class__.__name__
