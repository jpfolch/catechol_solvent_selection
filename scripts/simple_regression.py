import matplotlib.pyplot as plt
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)
import pandas as pd
from catechol.models import GPModel
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
from catechol.plots.plot_covariance_matrix import plot_gp_covariance_matrix
from catechol.data.featurizations import _load_featurization_lookup

# model = GPModel(featurization="spange_descriptors")
model = GPModel(multitask=True, featurization="spange_descriptors")
X, Y = load_single_solvent_data()
# remove unnecessary columns
X = X[INPUT_LABELS_SINGLE_SOLVENT]

# drop any solvents that can't be featurized
featurization_lookup = _load_featurization_lookup(model.featurization)
has_featurization = X["SOLVENT NAME"].isin(featurization_lookup.index)
X = X.loc[has_featurization]
Y = Y.loc[has_featurization]

# # remove all low-temperature solvents
# high_temperature_mask = (X["Temperature"] == 225) | (X["Temperature"] == 175)
high_temperature_mask = (X["Temperature"] == 225)
X = X[high_temperature_mask]
Y = Y[high_temperature_mask]

# you can also use leave-one-out splits of the data
split_generator = generate_leave_one_out_splits(X, Y)
# this will generate a new split each time you call `next` on the generator
# you can, instead, use a for loop to iterate over split_generator
next(split_generator)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
model.train(train_X, train_Y)

predictions = model.predict(test_X)

# calculate some metrics
mse = metrics.mse(predictions, test_Y)
nlpd = metrics.nlpd(predictions, test_Y)
print(f"{mse=}, {nlpd=}")

# plot the predictions
plot_solvent_prediction(model, test_X, test_Y)
_, (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)

# plot_gp_covariance_matrix(model, train_X)

plt.show()
