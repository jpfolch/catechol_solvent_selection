import numpy as np

from catechol.data.data_labels import INPUT_LABELS_FULL_DATA
from catechol.data.loader import (
    load_solvent_ramp_data,
    replace_repeated_measurements_with_average,
)


class BOBenchmark:
    """
    Class to load the data and query the BO benchmark.
    """

    def __init__(self, featurization, lambdas=[5, 1, 3, 1 / 20], **kwargs):
        super().__init__(**kwargs)
        X, Y = load_solvent_ramp_data()

        X = X[INPUT_LABELS_FULL_DATA]

        # replace repeated measurements with their average
        X, Y = replace_repeated_measurements_with_average(X, Y)

        self.featurization = featurization

        # define the search space
        self.X = X
        # define the objective function
        self.Y = Y
        self.l1 = lambdas[0]
        self.l2 = lambdas[1]
        self.l3 = lambdas[2]
        self.l4 = lambdas[3]

        self.objective_precomputed = False

    def objective_function(self, idx):
        if self.objective_precomputed:
            return self.objective_values[idx]
        else:
            self.objective_values = np.zeros(len(self.X))
            for i in range(len(self.X)):
                self.objective_values[i] = self._objective_function(i)
            self.objective_precomputed = True
            return self.objective_values[idx]

    def _objective_function(self, idx):
        # obtain the output data
        X = self.X.iloc[idx]
        Y = self.Y.iloc[idx]

        total_yield = Y["Product 2"] + Y["Product 3"]

        if total_yield == 0:
            selectivity = 0
        else:
            selectivity = Y["Product 2"] / total_yield

        residence_time = X["Residence Time"]
        temperature = (X["Temperature"] - 175) / (225 - 175)

        return (
            self.l1 * total_yield
            + self.l2 * selectivity
            - self.l3 * temperature
            - self.l4 * residence_time
        )

    def get_search_space(self):
        return self.X

    def get_optimum(self):
        if self.objective_precomputed:
            best_idx = np.argmax(self.objective_values)
            return self.X.iloc[best_idx], self.objective_values[best_idx]
        else:
            best_idx = np.argmax(
                [self._objective_function(i) for i in range(len(self.X))]
            )
            return self.X.iloc[best_idx], self._objective_function(best_idx)
