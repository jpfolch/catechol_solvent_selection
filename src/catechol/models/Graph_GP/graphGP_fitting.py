import tensorflow as tf
import gpflow
from .graph_kernels import SPKernel
from .models import Graph_GPR
from gpflow.utilities import print_summary

def graphGP_fit(Gs,
                Y,
                kernel=SPKernel(trainable_lengthscales=True,
                                trainable_variance=False,
                                trainable_alpha=True,
                                trainable_beta=True,
                                kernel_type="SP",
                                )):
    '''fit a GP model with given graphs,
    Args:
        Gs (tuple): graph dataset in form (graphs, scalar features)
        Y (array): corresponding y values of graph dataset
        kernel (class): Graph kernel, SP kernel used as default
    Return:
        m (class): trained graph GP model
    '''
    #
    m = Graph_GPR((Gs, Y), kernel=kernel)
    m.likelihood.variance = tf.constant([1e-6], dtype=tf.float64)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, compile=False)
    print_summary(m.kernel)
    return m