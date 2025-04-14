from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    """Base class for all models."""
    def __init__(self):
        """Initialize the model."""
        self.is_fitted = False

    def train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        """Train the model on the given data."""
        # TODO: normalize the data here
        self._train(train_X, train_Y)
        self.is_fitted = True

    @abstractmethod
    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        """Abstract method to train the model. Must be implemented by subclasses."""
        pass
    
    def predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the model."""
        return self._predict(test_X)
    
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