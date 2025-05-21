#!/usr/bin/env python3

# We use a modified version, that doesn't break with input transforms.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-Task GP models.

References

.. [Bonilla2007MTGP]
    E. Bonilla, K. Chai and C. Williams. Multi-task Gaussian Process Prediction.
    Advances in Neural Information Processing Systems 20, NeurIPS 2007.

.. [Swersky2013MTBO]
    K. Swersky, J. Snoek and R. Adams. Multi-Task Bayesian Optimization.
    Advances in Neural Information Processing Systems 26, NeurIPS 2013.

.. [Doucet2010sampl]
    A. Doucet. A Note on Efficient Conditional Simulation of Gaussian Distributions.
    http://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf,
    Apr 2010.

.. [Maddox2021bohdo]
    W. Maddox, M. Balandat, A. Wilson, and E. Bakshy. Bayesian Optimization with
    High-Dimensional Outputs. https://arxiv.org/abs/2106.12997, Jun 2021.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
    get_covar_module_with_dim_scaled_prior,
)
from botorch.posteriors.multitask import MultitaskGPPosterior
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.kernels.multitask_kernel import MultitaskKernel
from gpytorch.likelihoods.multitask_gaussian_likelihood import (
    MultitaskGaussianLikelihood,
)
from gpytorch.means import MultitaskMean
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.prior import Prior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import LogNormalPrior
from gpytorch.settings import detach_test_caches
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, pop_from_cache
from linear_operator.operators import (
    BatchRepeatLinearOperator,
    CatLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    RootLinearOperator,
    to_linear_operator,
)
from torch import Tensor


class KroneckerMultiTaskGP(ExactGP, GPyTorchModel, FantasizeMixin):
    """Multi-task GP with Kronecker structure, using an ICM kernel.

    This model assumes the "block design" case, i.e., it requires that all tasks
    are observed at all data points.

    For posterior sampling, this model uses Matheron's rule [Doucet2010sampl] to compute
    the posterior over all tasks as in [Maddox2021bohdo] by exploiting Kronecker
    structure.

    When a multi-fidelity model has Kronecker structure, this means there is one
    covariance kernel over the fidelity features (call it `K_f`) and another over
    the rest of the input parameters (call it `K_i`), and the resulting covariance
    across inputs and fidelities is given by the Kronecker product of the two
    covariance matrices. This is equivalent to saying the covariance between
    two input and feature pairs is given by

    K((parameter_1, fidelity_1), (parameter_2, fidelity_2))
        = K_f(fidelity_1, fidelity_2) * K_i(parameter_1, parameter_2).

    Then the covariance matrix of `n_i` parameters and `n_f` fidelities can be
    codified as a Kronecker product of an `n_i x n_i` matrix and an
    `n_f x n_f` matrix, which is far more parsimonious than specifying the
    whole `(n_i * n_f) x (n_i * n_f)` covariance matrix.

    Example:
        >>> train_X = torch.rand(10, 2)
        >>> train_Y = torch.cat([f_1(X), f_2(X)], dim=-1)
        >>> model = KroneckerMultiTaskGP(train_X, train_Y)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: MultitaskGaussianLikelihood | None = None,
        data_covar_module: Module | None = None,
        task_covar_prior: Prior | None = None,
        rank: int | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A `MultitaskGaussianLikelihood`. If omitted, uses a
                `MultitaskGaussianLikelihood` with a `GammaPrior(1.1, 0.05)`
                noise prior.
            data_covar_module: The module computing the covariance (Kernel) matrix
                in data space. If omitted, uses an `RBFKernel`.
            task_covar_prior : A Prior on the task covariance matrix. Must operate
                on p.s.d. matrices. A common prior for this is the `LKJ` prior. If
                omitted, uses `LKJCovariancePrior` with `eta` parameter as specified
                in the keyword arguments (if not specified, use `eta=1.5`).
            rank: The rank of the ICM kernel. If omitted, use a full rank kernel.
            kwargs: Additional arguments to override default settings of priors,
                including:
                - eta: The eta parameter on the default LKJ task_covar_prior.
                A value of 1.0 is uninformative, values <1.0 favor stronger
                correlations (in magnitude), correlations vanish as eta -> inf.
                - sd_prior: A scalar prior over nonnegative numbers, which is used
                for the default LKJCovariancePrior task_covar_prior.
                - likelihood_rank: The rank of the task covariance matrix to fit.
                Defaults to 0 (which corresponds to a diagonal covariance matrix).
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y, X=transformed_X)

        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._num_outputs = train_Y.shape[-1]
        batch_shape, ard_num_dims = train_X.shape[:-2], transformed_X.shape[-1]
        num_tasks = train_Y.shape[-1]

        if rank is None:
            rank = num_tasks
        if likelihood is None:
            noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
            likelihood = MultitaskGaussianLikelihood(
                num_tasks=num_tasks,
                batch_shape=batch_shape,
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior.mode,
                ),
                rank=kwargs.get("likelihood_rank", 0),
            )
        if task_covar_prior is None:
            task_covar_prior = LKJCovariancePrior(
                n=num_tasks,
                eta=torch.tensor(kwargs.get("eta", 1.5)).to(train_X),
                sd_prior=kwargs.get(
                    "sd_prior",
                    SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05),
                ),
            )
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = MultitaskMean(
            base_means=ConstantMean(batch_shape=batch_shape), num_tasks=num_tasks
        )
        if data_covar_module is None:
            data_covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=ard_num_dims,
                batch_shape=batch_shape,
            )
        else:
            data_covar_module = data_covar_module

        self.covar_module = MultitaskKernel(
            data_covar_module=data_covar_module,
            num_tasks=num_tasks,
            rank=rank,
            batch_shape=batch_shape,
            task_covar_prior=task_covar_prior,
        )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, X: Tensor) -> MultitaskMultivariateNormal:
        if self.training:
            X = self.transform_inputs(X)

        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    @property
    def _task_covar_matrix(self):
        res = self.covar_module.task_covar_module.covar_matrix
        if detach_test_caches.on():
            res = res.detach()
        return res

    @property
    @cached(name="train_full_covar")
    def train_full_covar(self):
        # train_x = self.transform_inputs(self.train_inputs[0])
        train_x = self.train_inputs[0]

        # construct Kxx \otimes Ktt
        train_full_covar = self.covar_module(train_x).evaluate_kernel()
        if detach_test_caches.on():
            train_full_covar = train_full_covar.detach()
        return train_full_covar

    @property
    @cached(name="predictive_mean_cache")
    def predictive_mean_cache(self):
        # train_x = self.transform_inputs(self.train_inputs[0])
        train_x = self.train_inputs[0]
        train_noise = self.likelihood._shaped_noise_covar(train_x.shape)
        if detach_test_caches.on():
            train_noise = train_noise.detach()

        train_diff = self.train_targets - self.mean_module(train_x)
        train_solve = (self.train_full_covar + train_noise).solve(
            train_diff.reshape(*train_diff.shape[:-2], -1)
        )
        if detach_test_caches.on():
            train_solve = train_solve.detach()

        return train_solve

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> MultitaskGPPosterior:
        self.eval()

        if posterior_transform is not None:
            # this could be very costly, disallow for now
            raise NotImplementedError(
                "Posterior transforms currently not supported for "
                f"{self.__class__.__name__}"
            )

        X = self.transform_inputs(X)
        # train_x = self.transform_inputs(self.train_inputs[0])
        train_x = self.train_inputs[0]

        # construct Ktt
        task_covar = self._task_covar_matrix
        task_rootlt = self._task_covar_matrix.root_decomposition(
            method="diagonalization"
        )
        task_root = task_rootlt.root
        if task_covar.batch_shape != X.shape[:-2]:
            task_covar = BatchRepeatLinearOperator(
                task_covar, batch_repeat=X.shape[:-2]
            )
            task_root = BatchRepeatLinearOperator(
                to_linear_operator(task_root), batch_repeat=X.shape[:-2]
            )

        task_covar_rootlt = RootLinearOperator(task_root)

        # construct RR' \approx Kxx
        data_data_covar = self.train_full_covar.linear_ops[0]
        # populate the diagonalziation caches for the root and inverse root
        # decomposition
        data_data_evals, data_data_evecs = data_data_covar.diagonalization()

        # pad the eigenvalue and eigenvectors with zeros if we are using lanczos
        if data_data_evecs.shape[-1] < data_data_evecs.shape[-2]:
            cols_to_add = data_data_evecs.shape[-2] - data_data_evecs.shape[-1]
            zero_evecs = torch.zeros(
                *data_data_evecs.shape[:-1],
                cols_to_add,
                dtype=data_data_evals.dtype,
                device=data_data_evals.device,
            )
            zero_evals = torch.zeros(
                *data_data_evecs.shape[:-2],
                cols_to_add,
                dtype=data_data_evals.dtype,
                device=data_data_evals.device,
            )
            data_data_evecs = CatLinearOperator(
                data_data_evecs,
                to_linear_operator(zero_evecs),
                dim=-1,
                output_device=data_data_evals.device,
            )
            data_data_evals = torch.cat((data_data_evals, zero_evals), dim=-1)

        # construct K_{xt, x}
        test_data_covar = self.covar_module.data_covar_module(X, train_x)
        # construct K_{xt, xt}
        test_test_covar = self.covar_module.data_covar_module(X)

        # now update root so that \tilde{R}\tilde{R}' \approx K_{(x,xt), (x,xt)}
        # cloning preserves the gradient history
        updated_linear_op = data_data_covar.cat_rows(
            cross_mat=test_data_covar.clone(),
            new_mat=test_test_covar,
            method="diagonalization",
        )
        updated_root = updated_linear_op.root_decomposition().root
        # occasionally, there's device errors so enforce this comes out right
        updated_root = updated_root.to(data_data_covar.device)

        # build a root decomposition of the joint train/test covariance matrix
        # construct (\tilde{R} \otimes M)(\tilde{R} \otimes M)' \approx
        # (K_{(x,xt), (x,xt)} \otimes Ktt)
        joint_covar = RootLinearOperator(
            KroneckerProductLinearOperator(
                updated_root, task_covar_rootlt.root.detach()
            )
        )

        # construct K_{xt, x} \otimes Ktt
        test_obs_kernel = KroneckerProductLinearOperator(test_data_covar, task_covar)

        # collect y - \mu(x) and \mu(X)
        train_diff = self.train_targets - self.mean_module(train_x)
        if detach_test_caches.on():
            train_diff = train_diff.detach()
        test_mean = self.mean_module(X)

        train_noise = self.likelihood._shaped_noise_covar(train_x.shape)
        diagonal_noise = isinstance(train_noise, DiagLinearOperator)
        if detach_test_caches.on():
            train_noise = train_noise.detach()
        test_noise = (
            self.likelihood._shaped_noise_covar(X.shape) if observation_noise else None
        )

        # predictive mean and variance for the mvn
        # first the predictive mean
        pred_mean = (
            test_obs_kernel.matmul(self.predictive_mean_cache).reshape_as(test_mean)
            + test_mean
        )
        # next the predictive variance, assume diagonal noise
        test_var_term = KroneckerProductLinearOperator(
            test_test_covar, task_covar
        ).diagonal()

        if diagonal_noise:
            task_evals, task_evecs = self._task_covar_matrix.diagonalization()
            # TODO: make this be the default KPMatmulLT diagonal method in gpytorch
            full_data_inv_evals = (
                KroneckerProductDiagLinearOperator(
                    DiagLinearOperator(data_data_evals), DiagLinearOperator(task_evals)
                )
                + train_noise
            ).inverse()
            test_train_hadamard = KroneckerProductLinearOperator(
                test_data_covar.matmul(data_data_evecs).to_dense() ** 2,
                task_covar.matmul(task_evecs).to_dense() ** 2,
            )
            data_var_term = test_train_hadamard.matmul(full_data_inv_evals).sum(dim=-1)
        else:
            # if non-diagonal noise (but still kronecker structured), we have to pull
            # across the noise because the inverse is not closed form
            # should be a kronecker lt, R = \Sigma_X^{-1/2} \kron \Sigma_T^{-1/2}
            # TODO: enforce the diagonalization to return a KPLT for all shapes in
            # gpytorch or dense linear algebra for small shapes
            data_noise, task_noise = train_noise.linear_ops
            data_noise_root = data_noise.root_inv_decomposition(
                method="diagonalization"
            )
            task_noise_root = task_noise.root_inv_decomposition(
                method="diagonalization"
            )

            # ultimately we need to compute the diagonal of
            # (K_{x* X} \kron K_T)(K_{XX} \kron K_T + \Sigma_X \kron \Sigma_T)^{-1}
            #                           (K_{x* X} \kron K_T)^T
            # = (K_{x* X} \Sigma_X^{-1/2} Q_R)(\Lambda_R + I)^{-1}
            #                       (K_{x* X} \Sigma_X^{-1/2} Q_R)^T
            # where R = (\Sigma_X^{-1/2T}K_{XX}\Sigma_X^{-1/2} \kron
            #                   \Sigma_T^{-1/2T}K_{T}\Sigma_T^{-1/2})
            # first we construct the components of R's eigen-decomposition
            # TODO: make this be the default KPMatmulLT diagonal method in gpytorch
            whitened_data_covar = (
                data_noise_root.transpose(-1, -2)
                .matmul(data_data_covar)
                .matmul(data_noise_root)
            )
            w_data_evals, w_data_evecs = whitened_data_covar.diagonalization()
            whitened_task_covar = (
                task_noise_root.transpose(-1, -2)
                .matmul(self._task_covar_matrix)
                .matmul(task_noise_root)
            )
            w_task_evals, w_task_evecs = whitened_task_covar.diagonalization()

            # we add one to the eigenvalues as above (not just for stability)
            full_data_inv_evals = (
                KroneckerProductDiagLinearOperator(
                    DiagLinearOperator(w_data_evals), DiagLinearOperator(w_task_evals)
                )
                .add_jitter(1.0)
                .inverse()
            )

            test_data_comp = (
                test_data_covar.matmul(data_noise_root).matmul(w_data_evecs).to_dense()
                ** 2
            )
            task_comp = (
                task_covar.matmul(task_noise_root).matmul(w_task_evecs).to_dense() ** 2
            )

            test_train_hadamard = KroneckerProductLinearOperator(
                test_data_comp, task_comp
            )
            data_var_term = test_train_hadamard.matmul(full_data_inv_evals).sum(dim=-1)

        pred_variance = test_var_term - data_var_term
        specialized_mvn = MultitaskMultivariateNormal(
            pred_mean, DiagLinearOperator(pred_variance)
        )
        if observation_noise:
            specialized_mvn = self.likelihood(specialized_mvn)

        posterior = MultitaskGPPosterior(
            distribution=specialized_mvn,
            joint_covariance_matrix=joint_covar,
            test_train_covar=test_obs_kernel,
            train_diff=train_diff,
            test_mean=test_mean,
            train_train_covar=self.train_full_covar,
            train_noise=train_noise,
            test_noise=test_noise,
        )

        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        return posterior

    def train(self, val=True, *args, **kwargs):
        if val:
            fixed_cache_names = ["data_data_roots", "train_full_covar", "task_root"]
            for name in fixed_cache_names:
                try:
                    pop_from_cache(self, name)
                except CachingError:
                    pass

        return super().train(val, *args, **kwargs)
