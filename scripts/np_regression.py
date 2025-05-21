from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)
from catechol.models.np import NPModel
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
from catechol.plots.plot_covariance_matrix import plot_gp_covariance_matrix
from catechol.data.featurizations import _load_featurization_lookup

import matplotlib.pyplot as plt

from torch.optim import Adam
import torch

torch.manual_seed(0)

#Architecture hyperparameters
h_dim = 64
r_dim = 32
z_dim = 8
sigma_bound = 0.01

# Training hyperparameters
optimizer = Adam
learning_rate = 0.001   
num_epochs = 5000
max_context_proportion = 0.3
min_context_proportion = 0.1

# Model 

model = NPModel(h_dim=h_dim, 
                r_dim = r_dim, 
                z_dim = z_dim, 
                optimizer = optimizer,
                learning_rate = learning_rate,
                num_epochs = num_epochs,
                max_context_proportion = max_context_proportion,
                min_context_proportion = min_context_proportion,
                sigma_bound = sigma_bound,)

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
(train_X, train_Y), (test_X, test_Y) = next(split_generator)

model.train(X, Y)

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

