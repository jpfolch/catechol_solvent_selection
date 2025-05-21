import copy

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.transforms.input import ChainedInputTransform, InputTransform, Warp
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior
from scipy.stats import norm

from catechol.data.data_labels import (
    get_data_labels_mean_var,
    is_df_solvent_ramp_dataset,
)
from catechol.data.featurizations import FeaturizationType, featurize_input_df
from catechol.models.learn_mean import LearnMean
from catechol.models.multitask import KroneckerMultiTaskGP

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
            raise ValueError(
                "Interpolating values must have the same number of dimensions."
            )

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


def get_separated_kernel(aug_batch_shape, d: int, cont_dims: list[int]):
    """Create a kernel that separately considers the continuous and featurized dimensions.

    This is useful for unseen solvents, where we still want to predict according
    to the residence time/temperature.
    """
    cont_kernel_factory = get_covar_module_with_dim_scaled_prior
    feat_dims = sorted(set(range(d)) - set(cont_dims))
    sum_kernel = ScaleKernel(
        cont_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(cont_dims),
            active_dims=cont_dims,
        )
        + cont_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(feat_dims),
            active_dims=feat_dims,
        )
    )
    prod_kernel = ScaleKernel(
        cont_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(cont_dims),
            active_dims=cont_dims,
        )
        * cont_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(feat_dims),
            active_dims=feat_dims,
        )
    )
    return sum_kernel + prod_kernel


class GPModel(Model):
    def __init__(
        self,
        multitask: bool = False,
        transfer_learning: bool = False,
        use_input_warp: bool = False,
        featurization: FeaturizationType | None = None,
        learn_prior_mean: bool = False,
        use_separated_kernel: bool = False,
        al_strategy: str = "mutual_information",
        bo_strategy: str = "ei",
    ):
        super().__init__(featurization=featurization)
        self.multitask = multitask
        self.use_input_warp = use_input_warp
        self.transfer_learning = transfer_learning
        self.active_learning_strategy = al_strategy
        self.learn_prior_mean = learn_prior_mean
        self.use_separated_kernel = use_separated_kernel
        self.bo_strategy = bo_strategy
        self.target_labels = []
        if transfer_learning:
            # we use the SM column to identify the task
            self.extra_input_columns = ["SM SMILES"]
            self.extra_input_columns_full = ["SM SMILES"]

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
            transforms.update(
                {
                    f"warp_{col}": Warp(
                        d,
                        [train_X_featurized.columns.get_loc(col)],
                        concentration0_prior=LogNormalPrior(0.0, 0.30**0.5),
                        concentration1_prior=LogNormalPrior(0.0, 0.30**0.5),
                        bounds=bounds,
                    )
                    for col in warp_cols
                }
            )

        if interpolate:
            alpha_dim = train_X_featurized.columns.get_loc("SolventB%")
            solvent_a_dims_bool = train_X_featurized.columns.str.startswith("A_")
            solvent_b_dims_bool = train_X_featurized.columns.str.startswith("B_")
            solvent_a_dims = np.argwhere(solvent_a_dims_bool).flatten().tolist()
            solvent_b_dims = np.argwhere(solvent_b_dims_bool).flatten().tolist()
            transforms["interp"] = InterpolationTransform(
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
        if self.bo_strategy == "ei":
            self.best_observation = train_Y.max().max()

        train_X_tensor = torch.tensor(
            train_X_featurized.to_numpy(), dtype=torch.float64
        )
        train_Y_tensor = torch.tensor(train_Y.to_numpy(), dtype=torch.float64)

        mean_module = (
            LearnMean(train_X, train_Y) if self.learn_prior_mean else ZeroMean()
        )

        interpolate = is_df_solvent_ramp_dataset(train_X)
        input_transform = self._get_input_transform(train_X_featurized, interpolate)

        if self.use_separated_kernel:
            # get shapes needed for defining the covariance function
            _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
                train_X=train_X_tensor, train_Y=train_Y_tensor
            )
            transformed_X = (
                input_transform.transform(train_X_tensor)
                if input_transform
                else train_X_tensor
            )
            d = transformed_X.shape[-1]
            covar_module = get_separated_kernel(aug_batch_shape, d, cont_dims=[0, 1])
        else:
            covar_module = None

        if self.transfer_learning:
            # we use an MTGP since only one experiment is observed for each X
            task_feature = train_X_featurized.columns.get_loc("SM SMILES")

            model = MultiTaskGP(
                train_X_tensor,
                train_Y_tensor,
                task_feature=task_feature,
                input_transform=input_transform,
                covar_module=covar_module,
            )
        else:
            model_cls = KroneckerMultiTaskGP if self.multitask else SingleTaskGP
            model = model_cls(
                train_X_tensor,
                train_Y_tensor,
                mean_module=mean_module,
                covar_module=covar_module,
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
            # we need to manually set the number of non_task_features for the MultiTaskGP
            # because of a bug in MultiTaskGP.posterior, where it assumes the task
            # feature is not included when the input transform changes the number of
            # dimensions
            if isinstance(self.model, MultiTaskGP):
                self.model.num_non_task_features = (
                    self.model.num_non_task_features - 1
                ) * 2 + 1
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
        learnmean = "-learnmean" if self.learn_prior_mean else ""
        separated = "-separated" if self.use_separated_kernel else ""
        return f"{self.__class__.__name__}{multi}{warp}{learnmean}{separated}"

    def select_next_ramp(
        self, ramps_to_train: list[int], all_ramps: list[int], X: pd.DataFrame
    ):
        """
        Select the next ramp to add to the training set. We use the entropy criterion.
        """
        # obtain a list of the ramps we can choose from
        ramps_to_test = [ramp for ramp in all_ramps if ramp not in ramps_to_train]

        if self.active_learning_strategy == "entropy":
            return self._select_next_ramp_entropy(X, ramps_to_test, all_ramps)

        elif self.active_learning_strategy == "random":
            # randomly select a ramp
            rng = np.random.default_rng()
            return rng.choice(ramps_to_test)

        elif self.active_learning_strategy == "mutual_information":
            return self._select_next_ramp_mutual_information(
                X, ramps_to_test, all_ramps
            )

    def _select_next_ramp_entropy(
        self, X: pd.DataFrame, ramps_to_test: list[str], all_ramps: list[str]
    ):
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

    def _select_next_ramp_mutual_information(
        self, X: pd.DataFrame, ramps_to_test: list[str], all_ramps: list[str]
    ):
        mutual_infos = []
        # loop over the ramps we can choose from

        for ramp_num in ramps_to_test:
            # first calculate the entropy of X_y | train_X
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
                entropy_Y_given_X_train = posterior.entropy().numpy()

            # now calculate the entropy of X_y | all_ramps \ ramp_num \ train_X

            # create a clone of the model
            model_clone = copy.deepcopy(self.model)

            # create new training data
            train_ramps_conjugate = [ramp for ramp in ramps_to_test if ramp != ramp_num]

            X_conjugate = X[X["RAMP NUM"].isin(train_ramps_conjugate)]

            X_conjugate_featurized = featurize_input_df(
                X_conjugate,
                self.featurization,
                remove_constant=True,
                normalize_feats=True,
            )

            train_X_tensor = torch.from_numpy(X_conjugate_featurized.to_numpy()).to(
                torch.float64
            )

            # create dummy Y values, since we are not using them
            dummy_Y = torch.zeros(
                size=(self.model.train_inputs[0].shape[0], train_X_tensor.shape[0]),
                dtype=torch.float64,
            )

            # replace the models training data with the new training data
            model_clone.set_train_data(
                model_clone.transform_inputs(train_X_tensor).unsqueeze(0),
                dummy_Y,
                strict=False,
            )
            # now calculate the entropy of X_y | all_ramps \ ramp_num
            model_clone.eval()
            with torch.no_grad():
                posterior = model_clone.posterior(test_X_tensor, observation_noise=True)
                entropy_Y_given_conjugate = posterior.entropy().numpy()

            # calculate the mutual information
            mutual_info = entropy_Y_given_X_train - entropy_Y_given_conjugate

            mutual_infos.append(mutual_info)

        # return the ramp with the highest mutual information
        max_mutual_info_index = np.argmax(mutual_infos)
        return ramps_to_test[max_mutual_info_index]

    def select_next_bo(self, train_idx, X: pd.DataFrame):
        if self.bo_strategy == "ei":
            return self._select_next_bo_ei(train_idx, X)
        elif self.bo_strategy == "random":
            # set of points to choose from
            test_idx = [i for i in X.index if i not in train_idx]
            rng = np.random.default_rng()
            return rng.choice(test_idx)
        elif self.bo_strategy == "ucb":
            return self._select_next_bo_ucb(train_idx, X)

    def _select_next_bo_ei(self, train_idx, X: pd.DataFrame):
        test_idx = [i for i in X.index if i not in train_idx]

        test_X = X.iloc[test_idx]

        preds = self.predict(test_X)
        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean = preds[mean_lbl].to_numpy()
        var = preds[var_lbl].to_numpy()

        # calculate the expected improvement
        z = (mean - self.best_observation) / np.sqrt(var)
        ei = (self.best_observation - mean) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

        # return the index of the point with the highest expected improvement
        max_ei_index = np.argmax(ei)
        return test_idx[max_ei_index]

    def _select_next_bo_ucb(self, train_idx, X: pd.DataFrame):
        test_idx = [i for i in X.index if i not in train_idx]

        test_X = X.iloc[test_idx]

        preds = self.predict(test_X)
        mean_lbl, var_lbl = get_data_labels_mean_var(self.target_labels)
        mean = preds[mean_lbl].to_numpy()
        var = preds[var_lbl].to_numpy()

        # calculate the upper confidence bound
        ucb = mean + 1.96 * np.sqrt(var)

        # return the index of the point with the highest ucb
        max_ucb_index = np.argmax(ucb)
        return test_idx[max_ucb_index]
