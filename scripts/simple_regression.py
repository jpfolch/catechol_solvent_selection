import matplotlib.pyplot as plt
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)
from catechol.models import GPModel
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction

model = GPModel(featurization="fragprints")
X, Y = load_single_solvent_data()
# remove unnecessary columns
X = X[INPUT_LABELS_SINGLE_SOLVENT]

# # remove all low-temperature solvents
# high_temperature_mask = X["Temperature"] == 225
# X = X[high_temperature_mask]
# Y = Y[high_temperature_mask]

# you can split the data into train/test sets
# however, this is naive as it ignores the time-series nature of the data
train_X, test_X = train_test_split(X, train_percentage=0.8, seed=1)
train_Y, test_Y = train_test_split(Y, train_percentage=0.8, seed=1)

model.train(train_X, train_Y)

predictions = model.predict(test_X)
print(predictions)

# you can also use leave-one-out splits of the data
split_generator = generate_leave_one_out_splits(X, Y)
# this will generate a new split each time you call `next` on the generator
# you can, instead, use a for loop to iterate over split_generator
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
model.train(train_X, train_Y)

predictions = model.predict(test_X)
print(predictions)

# calculate some metrics
mse = metrics.mse(predictions, test_Y)
nlpd = metrics.nlpd(predictions, test_Y)
print(f"{mse=}, {nlpd=}")

# plot the predictions
plot_solvent_prediction(model, test_X, test_Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)

plt.show()
