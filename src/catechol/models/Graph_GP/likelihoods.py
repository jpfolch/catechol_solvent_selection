from gpflow.likelihoods.scalar_continuous import Gaussian, Optional, ConstantOrFunction, Any, _lower_bound, sqrt, ParameterOrFunction, prepare_parameter_or_function, \
    TensorType, tf, inherit_check_shapes, logdensities, np, MeanAndVariance
from gpflow.utilities.parameter_or_function import Function

def evaluate_parameter_or_function(
    value: ParameterOrFunction,
    X: TensorType,
    *,
    lower_bound: Optional[float] = None,
) -> TensorType:
    if isinstance(value, Function):
        result = value(X)
        if lower_bound is not None:
            result = tf.maximum(result, lower_bound)
        return result
    else:
        return value
    

class GraphGaussian(Gaussian):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    def __init__(
        self,
        variance: Optional[ConstantOrFunction] = None,
        *,
        scale: Optional[ConstantOrFunction] = None,
        variance_lower_bound: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
        variance: The noise variance; must be greater than
            ``variance_lower_bound``. This is mutually exclusive with `scale`.
        scale: The noise scale; must be greater than
            ``sqrt(variance_lower_bound)``. This is mutually exclusive with `variance`.
        variance_lower_bound: The lower (exclusive) bound of ``variance``.
        kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        self.variance_lower_bound = _lower_bound(variance_lower_bound)
        self.scale_lower_bound = sqrt(self.variance_lower_bound)
        if scale is None:
            if variance is None:
                variance = 1.0
            self.variance: Optional[ParameterOrFunction] = prepare_parameter_or_function(
                variance, lower_bound=self.variance_lower_bound
            )
            self.scale: Optional[ParameterOrFunction] = None
        else:
            if variance is None:
                self.variance = None
                self.scale = prepare_parameter_or_function(
                    scale, lower_bound=self.scale_lower_bound
                )
            else:
                assert False, "Cannot set both `variance` and `scale`."

    def _variance(self, X: TensorType) -> tf.Tensor:
        if self.variance is not None:
            return evaluate_parameter_or_function(
                self.variance, X, lower_bound=self.variance_lower_bound
            )
        else:
            assert self.scale is not None  # For mypy.
            return (
                evaluate_parameter_or_function(self.scale, X, lower_bound=self.scale_lower_bound)
                ** 2
            )

    def variance_at(self, X: TensorType) -> tf.Tensor:
        variance = self._variance(X)
        # shape = tf.concat([tf.shape(X)[:-1], [1]], 0)]
        shape = tf.concat([[len(X)], [1]], 0)
        return tf.broadcast_to(variance, shape)

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.gaussian(Y, F, self._variance(X))

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:  # pylint: disable=R0201
        return tf.identity(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        shape = tf.shape(F)
        return tf.broadcast_to(self._variance(X), shape)

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        return tf.identity(Fmu), Fvar + self._variance(X)

    @inherit_check_shapes
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self._variance(X)), axis=-1)

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        variance = self._variance(X)
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / variance,
            axis=-1,
        )