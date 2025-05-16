import copy
import os
import random
import time

import numpy as np
import pandas as pd
import pkg_resources
import torch
import torch.nn as nn
from rxnfp.models import SmilesClassificationModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from catechol.data.data_labels import get_data_labels_mean_var
from catechol.data.loader import generate_leave_one_out_splits, train_test_split

from .base_model import Model


class LLMModel(Model):
    extra_input_columns = ["Reaction SMILES"]
    extra_input_columns_full = ["Reaction SMILES A", "Reaction SMILES B"]

    def __init__(
        self,
        pretrained_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        freeze_backbone: bool = False,
        learning_rate_backbone: float = 1e-5,
        learning_rate_head: float = 1e-4,
        dropout_backbone: float = 0.1,
        dropout_head: float = 0.1,
        use_pooler_output: bool = True,
        custom_head: nn.Module = None,
        max_length_padding: int = None,
        epochs: int = 10,
        time_limit: float = 10800,
        use_validation: str = None,
        batch_size: int = None,
        **kwargs,
    ):
        super().__init__(featurization="smiles")
        self._set_seed()
        self.model_name = pretrained_model_name
        self.freeze_backbone = freeze_backbone
        self.learning_rate_backbone = learning_rate_backbone
        self.learning_rate_head = learning_rate_head
        self.dropout_backbone = dropout_backbone
        self.dropout_head = dropout_head
        self.use_pooler_output = use_pooler_output
        self.custom_head = custom_head
        self.max_length_padding = max_length_padding
        self.epochs = epochs
        self.use_validation = use_validation
        self.batch_size = batch_size
        self.time_limit = time_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable parameters for sigmoid
        self.sigmoid_a = nn.Parameter(torch.tensor(1.0))
        self.sigmoid_b = nn.Parameter(torch.tensor(0.0))

        # To be set during training
        self.numerical_mean = None
        self.numerical_std = None
        self.train_losses = []
        self.val_losses = []
        self.optimizer = None
        self.loss_fn = None
        self.is_mixture = False

        # Set up backbone, tokenizer
        self._init_backbone_and_tokenizer()

    def _set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _parameterized_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.sigmoid_a * x + self.sigmoid_b)

    def _init_backbone_and_tokenizer(self):
        if "rxnfp" in self.model_name.lower():
            if self.model_name == "rxnfp-pretrained":
                model_path = pkg_resources.resource_filename(
                    "rxnfp", "models/transformers/bert_pretrained"
                )
            else:
                raise ValueError(f"Unknown rxnfp model: {self.model_name}")

            rxnfp_model = SmilesClassificationModel(
                "bert", model_path, use_cuda=(self.device.type == "cuda")
            )
            self.backbone = rxnfp_model.model.bert
            self.tokenizer = rxnfp_model.tokenizer

            if hasattr(self.backbone, "classifier"):
                del self.backbone.classifier

            self.backbone.config.hidden_dropout_prob = self.dropout_backbone
            self.backbone.config.attention_probs_dropout_prob = self.dropout_backbone
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            config.hidden_dropout_prob = self.dropout_backbone
            config.attention_probs_dropout_prob = self.dropout_backbone
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.backbone = AutoModel.from_pretrained(self.model_name, config=config)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _init_head_and_optimizer(self, output_size: int = 3):
        self.head = self.custom_head or self._build_default_head(
            output_size, 2, self.dropout_head
        ).to(self.device)
        self.loss_fn = nn.MSELoss()
        param_groups = [
            {
                "params": self.backbone.parameters(),
                "lr": self.learning_rate_backbone,
            },
            {
                "params": self.head.parameters(),
                "lr": self.learning_rate_head,
            },
        ]
        if self.is_mixture:
            param_groups.append(
                {
                    "params": [self.sigmoid_a, self.sigmoid_b],
                    "lr": self.learning_rate_head,  # Or a separate learning rate if you want
                }
            )
        self.optimizer = torch.optim.Adam(param_groups)

    def _build_default_head(
        self, output_size: int, num_extra_features: int, dropout_rate: float
    ):
        hidden_size = self.backbone.config.hidden_size
        return nn.Sequential(
            nn.Linear(hidden_size + num_extra_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),
            # nn.Sigmoid()
        )

    def _get_backbone_output(self, input_ids, attention_mask, use_pooler_output=True):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if (
            use_pooler_output
            and hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state[:, 0, :]

    def _full_model_prediction(self, *inputs):
        if len(inputs) == 3:  # Standard case
            input_ids, attention_mask, normalized_numerical = inputs
            output = self._get_backbone_output(
                input_ids, attention_mask, self.use_pooler_output
            )
        else:  # Mixture case
            ids_a, mask_a, ids_b, mask_b, pct_b, normalized_numerical = inputs
            out_a = self._get_backbone_output(ids_a, mask_a, self.use_pooler_output)
            out_b = self._get_backbone_output(ids_b, mask_b, self.use_pooler_output)

            weight_b = self._parameterized_sigmoid(pct_b)
            output = out_a * (1 - weight_b) + out_b * weight_b

        combined = torch.cat([output, normalized_numerical], dim=1)
        return self.head(combined)

    def tokenize_smiles(self, smiles_list):
        if self.max_length_padding:
            encoding = self.tokenizer(
                smiles_list,
                padding=True,
                truncation=True,
                max_length=self.max_length_padding,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                smiles_list, padding=True, truncation=True, return_tensors="pt"
            )
        return encoding["input_ids"], encoding["attention_mask"]

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
        if "Reaction SMILES A" in X.columns and "Reaction SMILES B" in X.columns:
            smiles_a = X["Reaction SMILES A"].tolist()
            smiles_b = X["Reaction SMILES B"].tolist()
            pct_b = (
                torch.tensor(X["SolventB%"].values, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )
            self.is_mixture = True
        else:
            smiles = X["Reaction SMILES"].tolist()
            self.is_mixture = False

        numerical_values = X[["Residence Time", "Temperature"]].values
        numerical_tensor = torch.tensor(numerical_values, dtype=torch.float32).to(
            self.device
        )

        if self.numerical_mean is None or self.numerical_std is None:
            self.numerical_mean = numerical_tensor.mean(dim=0, keepdim=True)
            self.numerical_std = numerical_tensor.std(dim=0, keepdim=True)

        normalized_numerical = self._normalize_numerical(numerical_tensor)

        if Y is not None:
            targets = torch.tensor(Y.values, dtype=torch.float32).to(self.device)
        else:
            targets = None

        if self.is_mixture:
            ids_a, mask_a = self.tokenize_smiles(smiles_a)
            ids_b, mask_b = self.tokenize_smiles(smiles_b)
            return (
                ids_a.to(self.device),
                mask_a.to(self.device),
                ids_b.to(self.device),
                mask_b.to(self.device),
                pct_b,
                normalized_numerical,
                targets,
            )
        else:
            input_ids, attention_mask = self.tokenize_smiles(smiles)
            return (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                normalized_numerical,
                targets,
            )

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

        train_tensors = self._prepare_training_tensors(train_X_split, train_Y_split)
        dataset = TensorDataset(*train_tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation set prep
        if validation:
            val_tensors = self._prepare_training_tensors(val_X, val_Y)
            *val_inputs, val_targets = val_tensors
            best_val_loss = float("inf")
            best_head_state, best_backbone_state = None, None

        self._init_head_and_optimizer(output_size=train_Y.shape[1])
        self.head.train()
        self.backbone.train()
        # Then your training loop:
        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                *inputs, targets = batch
                self.optimizer.zero_grad()
                preds = self._full_model_prediction(*inputs)
                loss = self.loss_fn(preds, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * targets.size(0)

            if validation:
                # Validation
                self.head.eval()
                self.backbone.eval()
                with torch.no_grad():
                    val_predictions = self._full_model_prediction(*val_inputs)
                    val_loss = self.loss_fn(val_predictions, val_targets)
                self.val_losses.append(val_loss)
                # Save best weights
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_head_state = copy.deepcopy(self.head.state_dict())
                    best_backbone_state = copy.deepcopy(self.backbone.state_dict())

                self.head.train()
                self.backbone.train()
                val_loss = f"{val_loss.item():.4f}"
            avg_loss = epoch_loss / len(loader.dataset)
            self.train_losses.append(avg_loss)
            print(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Validation Loss: {val_loss}"
            )
            if time.time() - start_time > self.time_limit:
                print(
                    f"Stopping training due to time limit ({time.time() - start_time} seconds) reached."
                )
                break
        if validation:
            # Load best model
            self.head.load_state_dict(best_head_state)
            self.backbone.load_state_dict(best_backbone_state)
            print(
                f"\nBest model selected based on validation loss: {best_val_loss:.4f}"
            )

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        self.head.eval()
        self.backbone.eval()

        *inputs, _ = self._prepare_training_tensors(test_X)

        with torch.no_grad():
            preds = self._full_model_prediction(*inputs)
            mean = preds.cpu().numpy()
            var = torch.zeros_like(preds).cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self) -> pd.DataFrame:
        """Ask the model for a candidate experiment, for Bayesian optimization."""
        return self._ask()

    def _get_model_name(self) -> str:
        pretrained = "-rxnfp" if "rxnfp" in self.model_name.lower() else "-chemberta"
        return f"{self.__class__.__name__}{pretrained}"
