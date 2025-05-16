from abc import ABC, abstractmethod

import pandas as pd

from catechol.data.featurizations import FeaturizationType
from catechol.data.normalize import normalize


class Model(ABC):
    """Base class for all models."""

    normalize_inputs = True
    extra_input_columns = []
    extra_input_columns_full = []

    def __init__(self, featurization: FeaturizationType | None = None):
        """Initialize the model."""
        self.is_fitted = False
        self.featurization = featurization
        self.target_labels = []

    def train(
        self, train_X: pd.DataFrame, train_Y: pd.DataFrame, *args, **kwargs
    ) -> None:
        """Train the model on the given data."""
        if self.normalize_inputs:
            train_X = normalize(train_X)
        self._train(train_X, train_Y, *args, **kwargs)
        self.is_fitted = True
        self.target_labels = train_Y.columns.to_list()

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

    def _ask(self) -> pd.DataFrame:
        """Abstract method to ask the model for a candidate experiment. Must be implemented by subclasses."""
        pass

    def get_model_name(self) -> str:
        return f"{self._get_model_name()}-{self.featurization}"

    def _get_model_name(self) -> str:
        return self.__class__.__name__

    def select_next_ramp(
        self, ramps_to_train: list[str], ramp_list: list[str], X: pd.DataFrame
    ) -> str:
        """
        Select the next ramp to train on based on the model's predictions.
        This method should be implemented by subclasses.
        """
        pass

    def select_next_bo(self, X: pd.DataFrame) -> int:
        """
        Select the next experiment to query based on the model's predictions.
        This method should be implemented by subclasses.
        """
        pass
