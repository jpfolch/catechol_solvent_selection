# TODO: replace this with actual data

from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import load_single_solvent_data
from catechol.models import GPModel

model = GPModel(featurization="acs_pca_descriptors")
train_X, train_Y = load_single_solvent_data()
# remove unnecessary columns
train_X = train_X[INPUT_LABELS_SINGLE_SOLVENT]

model.train(train_X, train_Y)

# test_X = pd.DataFrame({"x1": [0.15, 0.25], "x2": [0.51, 0.7]})
# predictions = model.predict(test_X)
# print(predictions)
