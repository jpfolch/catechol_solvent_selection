import matplotlib.pyplot as plt
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)
from catechol.models import GPModel
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction

model = GPModel(featurization="acs_pca_descriptors")
X, Y = load_single_solvent_data()
# remove unnecessary columns
X = X[INPUT_LABELS_SINGLE_SOLVENT]

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

plot_solvent_prediction(predictions, test_Y)
plt.show()
