import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from catechol.data.data_labels import get_data_labels_mean_var
from catechol.data.featurizations import featurize_input_df, FeaturizationType
from catechol.data.loader import generate_leave_one_out_splits, train_test_split

from .base_model import Model


class MLPModel(Model):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        epochs: int = 100,
        use_validation: str = None,
        batch_size: int = 32,
        custom_MLP=None,
        featurization: FeaturizationType | None = None,
    ):
        super().__init__(featurization=featurization)

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.use_validation = use_validation
        self.batch_size = batch_size
        self.custom_MLP = custom_MLP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed()

        # To be set during training
        self.numerical_mean = None
        self.numerical_std = None
        self.train_losses = []
        self.val_losses = []
        self.MLP = None
        self.optimizer = None
        self.loss_fn = None

    def _set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_MLP(self, output_size: int, num_features: int, dropout_rate: float):
        return nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),
        )

    def _init_MLP_and_optimizer(self, num_features):
        self.MLP = self.custom_MLP or self._build_MLP(3, num_features, self.dropout).to(
            self.device
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.MLP.parameters(), lr=self.learning_rate)

    def _normalize_numerical(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.numerical_mean) / (self.numerical_std + 1e-8)

    def _generate_train_split(self, train_X, train_Y):
        if self.use_validation == "leave_one_solvent_out":
            split_generator = generate_leave_one_out_splits(train_X, train_Y)
            (train_X_split, train_Y_split), (val_X, val_Y) = next(split_generator)
        else:
            train_X_split, val_X = train_test_split(
                train_X, train_percentage=0.8, seed=1
            )
            train_Y_split, val_Y = train_test_split(
                train_Y, train_percentage=0.8, seed=1
            )
        return train_X_split, train_Y_split, val_X, val_Y

    def _prepare_training_tensors(self, X: pd.DataFrame, Y: pd.DataFrame = None):
        # Creating featurization of the solvent
        X_input = featurize_input_df(X, self.featurization, remove_constant=True)

        # Numerical features
        numerical_tensor = torch.tensor(X_input.values, dtype=torch.float32).to(
            self.device
        )

        if self.numerical_mean is None or self.numerical_std is None:
            self.numerical_mean = numerical_tensor.mean(dim=0, keepdim=True)
            self.numerical_std = numerical_tensor.std(dim=0, keepdim=True)

        # if self.featurization_type == "acs_pca_descriptors":
        #    input_tensor = self._normalize_numerical(numerical_tensor)
        input_tensor = numerical_tensor
        if Y is not None:
            targets = torch.tensor(Y.values, dtype=torch.float32).to(self.device)
        else:
            targets = None

        return input_tensor, targets

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        val_loss = "NA"
        validation = False
        if self.use_validation:
            train_X_split, train_Y_split, val_X, val_Y = self._generate_train_split(
                train_X, train_Y
            )
            validation = True
        else:
            train_X_split, train_Y_split = train_X, train_Y

        # Training set prep
        if not self.batch_size:
            self.batch_size = train_X_split.shape[0]

        train_inputs, train_targets = self._prepare_training_tensors(
            train_X_split, train_Y_split
        )
        dataset = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation set prep
        if validation:
            val_inputs, val_targets = self._prepare_training_tensors(val_X, val_Y)
            best_val_loss = float("inf")
            best_MLP = None

        # Set MLP and optimizer
        num_features = train_inputs.shape[1]
        self._init_MLP_and_optimizer(num_features)
        self.MLP.train()

        # Then your training loop:
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_input, batch_targets = batch
                self.optimizer.zero_grad()
                preds = self.MLP(batch_input)
                loss = self.loss_fn(preds, batch_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_input)

            if validation:
                # Validation
                self.MLP.eval()
                with torch.no_grad():
                    val_predictions = self.MLP(val_inputs)
                    val_loss = self.loss_fn(val_predictions, val_targets)
                self.val_losses.append(val_loss.item())
                # Save best weights
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_MLP = copy.deepcopy(self.MLP.state_dict())

                self.MLP.train()

                val_loss = f"{val_loss.item():.4f}"
            avg_loss = epoch_loss / len(loader.dataset)
            self.train_losses.append(avg_loss)
            print(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Validation Loss: {val_loss}"
            )

        if validation:
            # Load best model
            self.MLP.load_state_dict(best_MLP)

            print(
                f"\nBest model selected based on validation loss: {best_val_loss:.4f}"
            )

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        self.MLP.eval()

        input_X, _ = self._prepare_training_tensors(test_X)

        with torch.no_grad():
            preds = self.MLP(input_X)
            mean = preds.cpu().numpy()
            var = torch.zeros_like(preds).cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var()
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self) -> pd.DataFrame:
        """Ask the model for a candidate experiment, for Bayesian optimization."""
        return self._ask()
