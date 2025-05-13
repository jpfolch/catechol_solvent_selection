import copy

import pandas as pd
import tensorflow as tf
from .Graph_GP.graphGP_fitting import graphGP_fit
from .Graph_GP.graph_kernels import GrakelKernel, SPKernel
import numpy as np
from .base_model import Model
from scripts.featurize_graphs import featurize_to_graph
from catechol.data.data_labels import get_data_labels_mean_var

class GraphGPModel(Model):
    def __init__(
            self,
            target_labels: list[str] = ["Product 2", "Product 3", "SM"],    # list of labels (allow single or multiple)
            kernel_name: str = "BoGrape_SP",    # Implemented kernels: "BoGrape_SP", "BoGrape_ESP", "RW", "SW", "WL"
    ):
        super().__init__()
        self.target_labels = target_labels
        self.kernel_name = kernel_name
        self.smiles_column = "SOLVENT SMILES"
        self.time_column = "Residence Time"
        self.temp_column = "Temperature"

    def scale(self, xs):
        xs_mean, xs_std = xs.mean(axis=0), xs.std(axis=0)
        X_ = tf.convert_to_tensor(xs)
        X = (X_ - xs_mean) / xs_std
        return X

    def _train(self, train_X: pd.DataFrame, train_Y: pd.DataFrame) -> None:
        Gs = [featurize_to_graph(smiles) for smiles in train_X[self.smiles_column].tolist()]
        Xs = train_X[[self.time_column, self.temp_column]].to_numpy()    # time and temperature scalar features
        X = self.scale(Xs)
        G_data = (Gs, X)

        # fit GP using graph kernel
        if self.kernel_name in ["BoGrape_SP", "BoGrape_ESP"]:
            kernel = SPKernel(exp_option=True if self.kernel_name[-3] == "E" else False,
                     trainable_lengthscales=True,
                     trainable_variance=True,
                     trainable_alpha=True,
                     trainable_beta=True,
                     kernel_type="SP",
                     )
        elif self.kernel_name in ["RW", "SW", "WL"]:
            kernel = GrakelKernel(grakel_kernel_name=self.kernel_name,
                                  trainable_lengthscales=True,
                                  trainable_variance=True,
                                  trainable_alpha=True,
                                  trainable_beta=True,
                                  )
        else:
            raise NotImplementedError

        self.model = []
        for single_label in self.target_labels:
            ys = train_Y[single_label].to_numpy()[..., None].astype("float64")
            Y = self.scale(ys)
            sub_kernel = copy.deepcopy(kernel)
            self.model.append(graphGP_fit(G_data, Y, sub_kernel))

    def _predict(self, test_X: pd.DataFrame) -> pd.DataFrame:
        Gs = [featurize_to_graph(smiles) for smiles in test_X[self.smiles_column].tolist()]
        Xs = test_X[[self.time_column, self.temp_column]].to_numpy()  # time and temperature scalar features
        X = self.scale(Xs)
        G_data = (Gs, X)

        means, vars = [], []
        for m in self.model:
            mean, var = m.predict_f(G_data)
            means.append(mean)
            vars.append(var)
        mean_lbl, var_lbl = get_data_labels_mean_var()
        mean_df = pd.DataFrame(tf.concat(means, axis=1), columns=mean_lbl)
        var_df = pd.DataFrame(tf.concat(vars, axis=1), columns=var_lbl)
        return pd.concat([mean_df, var_df], axis=1)


    def ask(self) -> pd.DataFrame:
        # TODO: implement BO for GP
        pass