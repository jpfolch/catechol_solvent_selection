import numpy as np
import pandas as pd
import torch
import copy
from botorch import fit_gpytorch_mll
from botorch.models import KroneckerMultiTaskGP, MultiTaskGP, SingleTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from catechol.data.data_labels import (
    get_data_labels_mean_var,
    is_df_solvent_ramp_dataset,
)
from catechol.data.featurizations import FeaturizationType, featurize_input_df

from .base_model import Model


class GPModel(Model):
    def __init__(
        self,
        multitask: bool = False,
        transfer_learning: bool = False,
        use_input_warp: bool = False,
        featurization: FeaturizationType | None = None,
        al_strategy: str = "mutual_information",
    ):
        super().__init__(featurization=featurization)
        self.multitask = multitask
        self.use_input_warp = use_input_warp
        self.transfer_learning = transfer_learning
        self.active_learning_strategy = al_strategy
        self.target_labels = []
        if transfer_learning:
            # we use the SM column to identify the task
            self.extra_input_columns = ["SM SMILES"]
            self.extra_input_columns_full = ["SM SMILES"]

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

    def _get_input_transform(self, train_X_featurized: pd.DataFrame):
        """Get the warping input transform."""
        if not self.use_input_warp:
            return None

        # We only want to warp the time and solvent mixture ratio
        warp_col_mask = train_X_featurized.columns.isin(("Residence Time", "SolventB%"))
        indices = np.argwhere(warp_col_mask).flatten().tolist()
        d = train_X_featurized.shape[-1]
        bounds = torch.tensor([[0.0] * d, [1.0] * d])

        return Warp(
            d,
            indices,
            concentration0_prior=LogNormalPrior(0.0, 0.30**0.5),
            concentration1_prior=LogNormalPrior(0.0, 0.30**0.5),
            bounds=bounds,
        )

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame, mean_module = ZeroMean()) -> None:
        train_X_featurized = featurize_input_df(
            train_X, self.featurization, remove_constant=True, normalize_feats=True
        )
        if is_df_solvent_ramp_dataset(train_X):
            train_X_featurized = self._get_mixed_solvent_representation(
                train_X_featurized
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

        warp = self._get_input_transform(train_X_featurized)
        if self.transfer_learning:
            # we use an MTGP since only one experiment is observed for each X
            task_feature = train_X_featurized.columns.get_loc("SM SMILES")
            model = MultiTaskGP(
                train_X_tensor,
                train_Y_tensor,
                task_feature=task_feature,
                input_transform=warp,
            )
        else:
            model_cls = KroneckerMultiTaskGP if self.multitask else SingleTaskGP
            model = model_cls(
                train_X_tensor,
                train_Y_tensor,
                mean_module=mean_module,
                input_transform=warp,
            )

        self.model = model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, optimizer_kwargs=dict(timeout_sec=30))

        self.target_labels = train_Y.columns.to_list()

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        test_X_featurized = featurize_input_df(
            test_X, self.featurization, remove_constant=True, normalize_feats=True
        )
        if is_df_solvent_ramp_dataset(test_X):
            test_X_featurized = self._get_mixed_solvent_representation(
                test_X_featurized
            )

        if self.transfer_learning:
            # encode the reaction using integers
            # identify reaction by the starting material
            test_X_featurized["SM SMILES"] = (
                test_X_featurized["SM SMILES"].astype("category").cat.codes
            )

        test_X_tensor = torch.from_numpy(test_X_featurized.to_numpy()).to(torch.float64)
        with torch.no_grad():
            preds = self.model.posterior(test_X_tensor)
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
        # transfer = "-transfer" if self.transfer_learning else ""
        return f"{self.__class__.__name__}{multi}{warp}"

    def select_next_ramp(
        self, ramps_to_train: list[int], all_ramps: list[int], X: pd.DataFrame):
        """
        Select the next ramp to add to the training set. We use the entropy criterion.
        """
        # obtain a list of the ramps we can choose from
        ramps_to_test = [
            ramp for ramp in all_ramps if ramp not in ramps_to_train
        ]

        if self.active_learning_strategy == "entropy":
            return self._select_next_ramp_entropy(X, ramps_to_test, all_ramps)

        elif self.active_learning_strategy == "random":
            # randomly select a ramp
            rng = np.random.default_rng()
            return rng.choice(ramps_to_test)

        elif self.active_learning_strategy == "mutual_information":
            return self._select_next_ramp_mutual_information(X, ramps_to_test, all_ramps)
    

    def _select_next_ramp_entropy(self, X: pd.DataFrame, ramps_to_test: list[str], all_ramps: list[str]):

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
            if is_df_solvent_ramp_dataset(X_ramp):
                X_ramp_featurized = self._get_mixed_solvent_representation(
                    X_ramp_featurized
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

    def _select_next_ramp_mutual_information(self, X: pd.DataFrame, ramps_to_test: list[str], all_ramps: list[str]):

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
            if is_df_solvent_ramp_dataset(X_ramp):
                X_ramp_featurized = self._get_mixed_solvent_representation(
                    X_ramp_featurized
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

            if is_df_solvent_ramp_dataset(X_conjugate):
                X_conjugate_featurized = self._get_mixed_solvent_representation(
                    X_conjugate_featurized
                )
            train_X_tensor = torch.from_numpy(
                X_conjugate_featurized.to_numpy()
            ).to(torch.float64)

            # create dummy Y values, since we are not using them
            dummy_Y = torch.zeros(size = (self.model.train_inputs[0].shape[0], train_X_tensor.shape[0]), dtype=torch.float64)

            # replace the models training data with the new training data
            model_clone.train_inputs = (train_X_tensor,)
            model_clone.train_targets = dummy_Y

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

