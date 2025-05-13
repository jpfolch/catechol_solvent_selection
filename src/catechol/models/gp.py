import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.transforms.input import Warp, InputTransform, ChainedInputTransform
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from catechol.models.multitask import KroneckerMultiTaskGP
from catechol.data.data_labels import (
    get_data_labels_mean_var,
    is_df_solvent_ramp_dataset,
)
from catechol.data.featurizations import FeaturizationType, featurize_input_df

from .base_model import Model


class InterpolationTransform(InputTransform):
    """A transform that interpolates between two (sets of) values."""

    def __init__(
        self,
        alpha_dim: int,
        value_a_dims: list[int],
        value_b_dims: list[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ):
        super().__init__()
        if len(value_a_dims) != len(value_b_dims):
            raise ValueError("Interpolating values must have the same number of dimensions.")
        
        self.alpha_dim = alpha_dim
        self.value_a_dims = value_a_dims
        self.value_b_dims = value_b_dims

        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

        
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        alpha = X[..., [self.alpha_dim]]
        A = X[..., self.value_a_dims]
        B = X[..., self.value_b_dims]
        interp = A * (1 - alpha) + B * alpha

        dims_for_interp = [self.alpha_dim, *self.value_a_dims, *self.value_b_dims]
        remaining_dims = [d for d in range(X.shape[-1]) if d not in dims_for_interp]
        remaining_X = X[..., remaining_dims]

        interp_X = torch.cat([remaining_X, interp], dim=-1)
        return interp_X


class GPModel(Model):
    def __init__(
        self,
        multitask: bool = False,
        transfer_learning: bool = False,
        use_input_warp: bool = False,
        featurization: FeaturizationType | None = None,
    ):
        super().__init__(featurization=featurization)
        self.multitask = multitask
        self.use_input_warp = use_input_warp
        self.transfer_learning = transfer_learning
        self.target_labels = []
        if transfer_learning:
            # we use the SM column to identify the task
            self.extra_input_columns = ["SM SMILES"]

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

        any_featurized = X_featurized.columns.str.match("^(A_|B_)")
        return pd.concat(
            (
                X_featurized.loc[:, ~any_featurized],
                mixed_feat,
            ),
            axis="columns",
        )

    def _get_input_transform(self, train_X_featurized: pd.DataFrame, interpolate: bool):
        """Get the warping input transform."""
        transforms = {}
        if self.use_input_warp:
            # We only want to warp the time and solvent mixture ratio
            # warp_col_mask = train_X_featurized.columns.isin(("Residence Time", "SolventB%"))
            # indices = np.argwhere(warp_col_mask).flatten().tolist()
            d = train_X_featurized.shape[-1]
            bounds = torch.tensor([[0.0] * d, [1.0] * d])

            # we have to do the warps independently for each of the columns
            # https://github.com/pytorch/botorch/issues/2852
            warp_cols = ["Residence Time"] + ["SolventB%"] * interpolate
            transforms.update({
                f"warp_{col}": Warp(
                d,
                [train_X_featurized.columns.get_loc(col)],
                concentration0_prior=LogNormalPrior(0.0, 0.30**0.5),
                concentration1_prior=LogNormalPrior(0.0, 0.30**0.5),
                bounds=bounds,
            ) for col in warp_cols})

        if interpolate:
            alpha_dim = train_X_featurized.columns.get_loc("SolventB%")
            solvent_a_dims_bool = train_X_featurized.columns.str.startswith(f"A_")
            solvent_b_dims_bool = train_X_featurized.columns.str.startswith(f"B_")
            solvent_a_dims = np.argwhere(solvent_a_dims_bool).flatten().tolist()
            solvent_b_dims = np.argwhere(solvent_b_dims_bool).flatten().tolist()
            transforms["interp"]  = InterpolationTransform(
                alpha_dim,
                solvent_a_dims,
                solvent_b_dims,                
            )

        return ChainedInputTransform(**transforms) if transforms else None

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        train_X_featurized = featurize_input_df(
            train_X, self.featurization, remove_constant=True, normalize_feats=True
        )

        if self.transfer_learning:
            # encode the reaction using integers
            # identify reaction by the starting material
            train_X_featurized["SM SMILES"] = (
                train_X_featurized["SM SMILES"].astype("category").cat.codes
            )

        train_X_tensor = torch.tensor(
            train_X_featurized.to_numpy(), dtype=torch.float64
        )
        train_Y_tensor = torch.tensor(train_Y.to_numpy(), dtype=torch.float64)

        interpolate = is_df_solvent_ramp_dataset(train_X)
        input_transform = self._get_input_transform(train_X_featurized, interpolate)

        if self.transfer_learning:
            # we use an MTGP since only one experiment is observed for each X
            task_feature = train_X_featurized.columns.get_loc("SM SMILES")
            model = MultiTaskGP(
                train_X_tensor,
                train_Y_tensor,
                task_feature=task_feature,
                input_transform=input_transform,
            )
        else:
            model_cls = KroneckerMultiTaskGP if self.multitask else SingleTaskGP
            model = model_cls(
                train_X_tensor,
                train_Y_tensor,
                mean_module=ZeroMean(),
                input_transform=input_transform,
            )

        self.model = model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, optimizer_kwargs=dict(timeout_sec=30))

        self.target_labels = train_Y.columns.to_list()

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        test_X_featurized = featurize_input_df(
            test_X, self.featurization, remove_constant=True, normalize_feats=True
        )
        
        if self.transfer_learning:
            # encode the reaction using integers
            # identify reaction by the starting material
            test_X_featurized["SM SMILES"] = (
                test_X_featurized["SM SMILES"].astype("category").cat.codes
            )

        test_X_tensor = torch.from_numpy(test_X_featurized.to_numpy()).to(torch.float64)
        with torch.no_grad():
            preds = self.model.posterior(test_X_tensor, observation_noise=True)
            mean = preds.mean.cpu().numpy()
            var = preds.variance.cpu().numpy()

        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)

    def _ask(self):
        # TODO: implement BO for GP
        pass

    def _get_model_name(self) -> str:
        multi = "-multi" if self.multitask else "-indep"
        warp = "-warp" if self.use_input_warp else ""
        transfer = "-transfer" if self.transfer_learning else ""
        return f"{self.__class__.__name__}{multi}{warp}{transfer}"

    def select_next_ramp(
        self, ramps_to_train: list[str], all_ramps: list[str], X: pd.DataFrame
    ):
        """
        Select the next ramp to add to the training set. We use the entropy criterion.
        """
        # obtain a list of the ramps we can choose from
        ramps_to_test = [
            ramp for ramp in all_ramps if ramp not in ramps_to_train
        ]

        entropies = []
        # loop over the ramps we can choose from
        for ramp_num in ramps_to_test:
            X_ramp = X[X["RAMP NUM"] == ramp_num]

            X_ramp_featurized = featurize_input_df(
                X_ramp,
                self.featurization,
                remove_constant=True,
                normalize_feats=True,
            )

            test_X_tensor = torch.from_numpy(X_ramp_featurized.to_numpy()).to(
                torch.float64
            )
            with torch.no_grad():
                posterior = self.model.posterior(test_X_tensor, observation_noise=True)
                entropy = posterior.entropy().numpy()

            entropies.append(entropy)

        # return the ramp with the highest entropy
        max_entropy_index = np.argmax(entropies)
        return ramps_to_test[max_entropy_index]
