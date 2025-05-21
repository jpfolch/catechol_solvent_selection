import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from catechol.data.data_labels import (
    get_data_labels_mean_var,
    is_df_solvent_ramp_dataset,
)
from catechol.data.featurizations import FeaturizationType, featurize_input_df
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
        self.is_mixture = None

        # If mixture using sigmoid
        self.sigmoid_a = nn.Parameter(torch.tensor(1.0))
        self.sigmoid_b = nn.Parameter(torch.tensor(0.0))

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

    def _init_MLP_and_optimizer(self, num_features, output_size=3):
        self.MLP = self.custom_MLP or self._build_MLP(
            output_size, num_features, self.dropout
        ).to(self.device)
        self.loss_fn = nn.MSELoss()
        params = [
            {"params": self.MLP.parameters(), "lr": self.learning_rate},
        ]

        if self.is_mixture:
            params.append(
                {"params": [self.sigmoid_a, self.sigmoid_b], "lr": self.learning_rate}
            )  # or another LR

        self.optimizer = torch.optim.Adam(params)

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

    def _get_mixed_solvent_representation(self, X_featurized: pd.DataFrame):
        alpha = X_featurized["SolventB%"]

        def get_solvent_feat(solvent: str):
            feat = X_featurized.loc[
                :, X_featurized.columns.str.startswith(f"{solvent}_")
            ]
            feat = feat.rename(columns=lambda c: c.removeprefix(f"{solvent}_"))
            return feat

        A_feat = get_solvent_feat("A")
        B_feat = get_solvent_feat("B")

        mixed_feat = A_feat.mul(1 - alpha, axis=0) + B_feat.mul(alpha, axis=0)
        return pd.concat(
            (
                X_featurized[["Residence Time", "Temperature"]],
                mixed_feat,
            ),
            axis="columns",
        )

    def _prepare_training_tensors(self, X: pd.DataFrame, Y: pd.DataFrame = None):
        # Creating featurization of the solvent
        X_input = featurize_input_df(X, self.featurization, remove_constant=True)

        # Numerical
        numerical_values = X_input[["Residence Time", "Temperature"]].values
        numerical_tensor = torch.tensor(numerical_values, dtype=torch.float32).to(
            self.device
        )
        if self.numerical_mean is None or self.numerical_std is None:
            self.numerical_mean = numerical_tensor.mean(dim=0, keepdim=True)
            self.numerical_std = numerical_tensor.std(dim=0, keepdim=True)
        numerical_tensor = (numerical_tensor - self.numerical_mean) / self.numerical_std

        if is_df_solvent_ramp_dataset(X):
            self.is_mixture = True
            # X_input = self._get_mixed_solvent_representation(X_input)
            pct_B = (
                torch.tensor(X["SolventB%"].values, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1)
            )

            def get_solvent_feat_tensor(prefix):
                cols = [c for c in X_input.columns if c.startswith(f"{prefix}_")]
                tensor = torch.tensor(X_input[cols].values, dtype=torch.float32).to(
                    self.device
                )
                return tensor

            A_feat_tensor = get_solvent_feat_tensor("A")
            B_feat_tensor = get_solvent_feat_tensor("B")

            input_tensor = torch.cat(
                [numerical_tensor, pct_B, A_feat_tensor, B_feat_tensor], dim=1
            )

        else:
            self.is_mixture = False
            featurization_cols = [
                col
                for col in X_input.columns
                if col not in ["Residence Time", "Temperature"]
            ]
            featurization_tensor = torch.tensor(
                X_input[featurization_cols].values, dtype=torch.float32
            ).to(self.device)
            input_tensor = torch.cat([numerical_tensor, featurization_tensor], dim=1)
        # if self.featurization == "acs_pca_descriptors":
        #     non_numerical_tensor = (non_numerical_tensor - non_numerical_tensor.mean(dim=0, keepdim=True)) / non_numerical_tensor.std(dim=0, keepdim=True)

        if Y is not None:
            targets = torch.tensor(Y.values, dtype=torch.float32).to(self.device)
        else:
            targets = None

        return input_tensor, targets

    def _full_model_prediction(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.is_mixture:
            return self.MLP(inputs)

        # Expecting inputs as: [Residence Time, Temperature, SolventB%, A_feat..., B_feat...]
        numerical = inputs[:, :2]
        pct_B = inputs[:, 2:3]
        num_feats = (inputs.shape[1] - 3) // 2
        A_feat = inputs[:, 3 : 3 + num_feats]
        B_feat = inputs[:, 3 + num_feats :]

        alpha = torch.sigmoid(self.sigmoid_a * pct_B + self.sigmoid_b)
        mixed = (1 - alpha) * A_feat + alpha * B_feat
        full_input = torch.cat([numerical, mixed], dim=1)

        return self.MLP(full_input)

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
        if self.is_mixture:
            num_features = int((train_inputs.shape[1] - 3) / 2 + 2)
        else:
            num_features = train_inputs.shape[1]
        num_outputs = train_targets.shape[1]
        self._init_MLP_and_optimizer(num_features, num_outputs)
        self.MLP.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_input, batch_targets = batch
                self.optimizer.zero_grad()
                preds = self._full_model_prediction(batch_input)
                loss = self.loss_fn(preds, batch_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_input)

            if validation:
                # Validation
                self.MLP.eval()
                with torch.no_grad():
                    val_predictions = self._full_model_prediction(val_inputs)
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
            preds = self._full_model_prediction(input_X)
            mean = preds.cpu().numpy()
            var = torch.zeros_like(preds).cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self) -> pd.DataFrame:
        """Ask the model for a candidate experiment, for Bayesian optimization."""
        return self._ask()
