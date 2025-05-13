import numpy as np
import tensorflow as tf
import gpflow
from gpflow.base import Parameter
from gpflow.kernels import Stationary, IsotropicStationary
from gpflow.utilities import positive, to_default_float
import tensorflow_probability as tfp
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, RandomWalkLabeled, SubgraphMatching, ShortestPath
from grakel import Graph as GKgraph

class SPKernel(IsotropicStationary):
    def __init__(self,
                 exp_option: bool = True,
                 variance: float = 1,
                 alpha: float = 1,
                 beta: float = 1,
                 lengthscales: float = 1,
                 kernel_type: str = "SP",
                 trainable_variance: bool = True,
                 trainable_alpha: bool = True,
                 trainable_beta: bool = True,
                 trainable_lengthscales: bool = True,
                 **kwargs):
        """
        Args:
            variance (float): kernel variance for exponential kernels (ESSP and ESP)
            alpha, beta (float): kernel trainable parameters
            lengthscales (float): default to 1
            kernel_type (str): kernel type "SSP" or "SP"
            trainable_variance (boolean): set to True for exponential kernels
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")
        KERNEL_PRIOR_SCALE = tf.constant(1.0, dtype=gpflow.default_float())

        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=tfp.bijectors.SoftClip(
                    to_default_float(0.01),
                    to_default_float(100),
                )) if trainable_variance else variance
        self.lengthscales = Parameter(lengthscales, transform=tfp.bijectors.SoftClip(
            to_default_float(0.01),
            to_default_float(100),
        )) if trainable_lengthscales else lengthscales
        self.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(self.lengthscales), KERNEL_PRIOR_SCALE
        )
        self.lengthscales2 = Parameter(lengthscales, transform=tfp.bijectors.SoftClip(
            to_default_float(0.01),
            to_default_float(100),
        )) if trainable_lengthscales else lengthscales
        self.lengthscales2.prior = tfp.distributions.LogNormal(
            tf.math.log(self.lengthscales2), KERNEL_PRIOR_SCALE
        )
        self.alpha = Parameter(alpha, transform=tfp.bijectors.SoftClip(
                    to_default_float(0.01),
                    to_default_float(100),
                )) if trainable_alpha else alpha
        self.beta = Parameter(beta, transform=tfp.bijectors.SoftClip(
                    to_default_float(0.01),
                    to_default_float(100),
                )) if trainable_beta else beta
        self.kernel_type = kernel_type
        self.exp_option = exp_option
        self._validate_ard_active_dims(self.lengthscales)

        super(Stationary).__init__(**kwargs)

    def _extract_graph_feature(self, G):

        return np.array(G.S)
         # return tf.convert_to_tensor(G.S)

    def _get_V_Matrix(self, Gs):
        '''calculate feature matrix for given graph dataset
        Args:
            G_data: graph dataset, size n
        Returns:
            V_matrix: shape (n x L+F+1)
        '''
        T = len(Gs)
        V_matrix = np.zeros(shape=(T, Gs[0].M))
        for G_id, G in enumerate(Gs):
            V_matrix[G_id] = self._extract_graph_feature(G)
        return tf.convert_to_tensor(V_matrix, dtype=tf.float64)

    def calculate_path_covariance(self, G1, G2):
        ans_P = 0.
        if self.kernel_type == "SSP":
            for l in range(min(G1.N, G2.N)):
                ans_P += G1.D[l] * G2.D[l]
        elif self.kernel_type == "SP":
            for l in range(min(G1.N, G2.N)):
                for l1 in range(G1.L):
                    for l2 in range(G2.L):
                        ans_P += G1.P[(l, l1, l2)] * G2.P[(l, l1, l2)]
        else:
            NotImplementedError("Graph kernel not supported.")
        ans_P /= ((G1.N ** 2) * (G2.N ** 2))
        return ans_P

    def calculate_node_feature_covariance(self, G1, G2):
        ans_X = 0.
        for i in range(G1.M):
            ans_X += G1.S[i] * G2.S[i] / (G1.N * G2.N * G1.M)
        return ans_X

    def calculate_scalar_covariance(self, X1, X2):
        # RBF kernel on scalar features
        # ans_F = tf.exp(-tf.linalg.norm(X1 - X2) ** 2 / (2 * tf.cast(self.lengthscales, dtype=tf.float64) ** 2))
        ans_F1 = -tf.linalg.norm(X1[0] - X2[0]) ** 2
        ans_F2 = -tf.linalg.norm(X1[1] - X2[1]) ** 2
        return ans_F1, ans_F2

    def K(self, KP, KX, KF1, KF2):
        K = tf.cast(self.alpha, dtype=tf.float64) * KP + tf.cast(self.beta, dtype=tf.float64) * KX
        if self.exp_option:
            K = tf.exp(K)
        K = self.variance * K \
              * tf.exp(KF1/ (2 * tf.cast(self.lengthscales, dtype=tf.float64) ** 2) + KF2/ (2 * tf.cast(self.lengthscales2, dtype=tf.float64) ** 2))
        return K

    def extract_kernel_matrix(self, G_data1, G_data2=None):
        G_data2 = G_data1 if G_data2 is None else G_data2
        Gs1, Xs1 = G_data1
        Gs2, Xs2 = G_data2
        m, n = len(Gs1), len(Gs2)
        KP = np.zeros(shape=(m, n))
        KX = np.zeros(shape=(m, n))
        KF1 = np.zeros(shape=(m, n))
        KF2 = np.zeros(shape=(m, n))
        for idx1 in range(m):
            for idx2 in range(n):
                G1, X1 = Gs1[idx1], Xs1[idx1]
                G2, X2 = Gs2[idx2], Xs2[idx2]
                ans_P = self.calculate_path_covariance(G1, G2)
                ans_X = self.calculate_node_feature_covariance(G1, G2)
                ans_F = self.calculate_scalar_covariance(X1, X2)
                KP[idx1, idx2] = ans_P
                KX[idx1, idx2] = ans_X
                KF1[idx1, idx2], KF2[idx1, idx2] = ans_F
        return tf.convert_to_tensor(KP, dtype=tf.float64), tf.convert_to_tensor(KX, dtype=tf.float64), tf.convert_to_tensor(KF1, dtype=tf.float64), tf.convert_to_tensor(KF2, dtype=tf.float64)

class GrakelKernel(SPKernel):
    def __init__(self, grakel_kernel_name, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = grakel_kernel_name
        if self.__name__ == "RW":
            self.grakel_kernel = RandomWalkLabeled(lamda=0.001, normalize=True, method_type="fast")
        elif self.__name__ == "SM":
            self.grakel_kernel = SubgraphMatching(k=5, normalize=True, ke=None)
        elif self.__name__ == "WL":
            self.grakel_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)
        else:
            print("Not implemented grakel kernel!")

    def transform_graphs(self, G):
        # From our graph to grakel graph
        adj = np.array(G.A) - np.eye(G.N)
        edge_num = int(np.sum(adj) / 2)
        adj = adj.tolist()
        node_labels = {}
        for v in range(G.N):
            node_labels[v] = G.l[v]
        if self.__name__ == "SM":
            edge_labels = {}
            for v in range(edge_num):
                edge_labels[v] = "a"
            grakel_graph = GKgraph(initialization_object=adj, node_labels=node_labels, edge_labels=edge_labels)
        else:
            grakel_graph = GKgraph(initialization_object=adj, node_labels=node_labels)
        return grakel_graph


    def extract_kernel_matrix(self, G_data1, G_data2=None):
        if G_data2 is None:
            Gs1, Xs1 = G_data1
            trans_G_data1 = [self.transform_graphs(g) for g in Gs1]
            KP = self.grakel_kernel.fit_transform(trans_G_data1)
            KP = KP.T
            n = len(Gs1)
            KX = np.zeros(shape=(n, n))
            KF1 = np.zeros(shape=(n, n))
            KF2 = np.zeros(shape=(n, n))
            for idx1, G1 in enumerate(Gs1):
                for idx2, G2 in enumerate(Gs1):
                    ans_X = self.calculate_node_feature_covariance(G1, G2)
                    ans_F = self.calculate_scalar_covariance(Xs1[idx1], Xs1[idx2])
                    KX[idx1, idx2] = ans_X
                    KF1[idx1, idx2], KF2[idx1, idx2] = ans_F

        else:
            Gs1, Xs1 = G_data1
            Gs2, Xs2 = G_data2
            trans_G_data1 = [self.transform_graphs(g) for g in Gs1]
            trans_G_data2 = [self.transform_graphs(g) for g in Gs2]
            self.grakel_kernel.fit_transform(trans_G_data1)
            KP = self.grakel_kernel.transform(trans_G_data2)
            KP = KP.T
            m, n = len(Gs1), len(Gs2)
            KX = np.zeros(shape=(m, n))
            KF1 = np.zeros(shape=(m, n))
            KF2 = np.zeros(shape=(m, n))
            for idx1, G1 in enumerate(Gs1):
                for idx2, G2 in enumerate(Gs2):
                    ans_X = self.calculate_node_feature_covariance(G1, G2)
                    ans_F = self.calculate_scalar_covariance(Xs1[idx1], Xs2[idx2])
                    KX[idx1, idx2] = ans_X
                    KF1[idx1, idx2], KF2[idx1, idx2] = ans_F

        return tf.convert_to_tensor(KP, dtype=tf.float64), tf.convert_to_tensor(KX, dtype=tf.float64), tf.convert_to_tensor(KF1, dtype=tf.float64), tf.convert_to_tensor(KF2, dtype=tf.float64)
