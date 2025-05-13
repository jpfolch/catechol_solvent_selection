import gpflow
import tensorflow as tf
from gpflow.models.gpr import RegressionData, Kernel, Optional, MeanFunction, TensorData,\
      Gaussian, add_likelihood_noise_cov, GPModel, InternalDataTrainingLossMixin, multivariate_normal, \
        InputData, MeanAndVariance, assert_params_false
import numpy as np


class Graph_GPR(GPModel, InternalDataTrainingLossMixin):
    def __init__(
            self,
            data: RegressionData,
            kernel: Kernel,
            mean_function: Optional[MeanFunction] = lambda Graph_List: tf.zeros(shape=(len(Graph_List), 1), dtype=tf.float64),
            noise_variance: Optional[TensorData] = None,
            likelihood: Optional[Gaussian] = None,
    ):
        assert (noise_variance is None) or (
                likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data
        self.prepare_kernel_matrix()

    def prepare_kernel_matrix(self):
        G_data, Y = self.data
        self.KP, self.KX, self.KF1, self.KF2 = self.kernel.extract_kernel_matrix(G_data)

    # type-ignore is because of changed method signature:
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r'''
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        '''
        G_data, Y = self.data
        Gs, X = G_data
        X_ = self.kernel._get_V_Matrix(Gs)
        K = self.kernel.K(self.KP, self.KX, self.KF1, self.KF2)
        ks = add_likelihood_noise_cov(K, self.likelihood, X_)

        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X_)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
            self, Gnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r'''
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        '''

        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        G_data, Y = self.data
        Gs, X = G_data
        Gsnew, Xnew = Gnew
        X_ = self.kernel._get_V_Matrix(Gs)
        Xnew_ = self.kernel._get_V_Matrix(Gsnew)
        err = Y - self.mean_function(X_)

        kmm = self.kernel.K(self.KP, self.KX, self.KF1, self.KF2)
        KPnn, KXnn, KFnn1, KFnn2 = self.kernel.extract_kernel_matrix(Gnew)
        knn = self.kernel.K(KPnn, KXnn, KFnn1, KFnn2)
        if not full_cov:
            knn = tf.linalg.diag_part(knn)
        KPmn, KXmn, KFmn1, KFmn2 = self.kernel.extract_kernel_matrix(G_data, Gnew)
        kmn = self.kernel.K(KPmn, KXmn, KFmn1, KFmn2)
        kmm_plus_s = add_likelihood_noise_cov(kmm, self.likelihood, X_)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )
        f_mean = f_mean_zero + self.mean_function(Xnew_)
        return f_mean, f_var