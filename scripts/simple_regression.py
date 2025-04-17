from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import load_single_solvent_data, train_test_split
from catechol.models import GPModel

model = GPModel(featurization="acs_pca_descriptors")
train_X, train_Y = load_single_solvent_data()
# remove unnecessary columns
train_X = train_X[INPUT_LABELS_SINGLE_SOLVENT]

train_X, test_X = train_test_split(train_X, train_percentage=0.8, seed=1)
train_Y, test_Y = train_test_split(train_Y, train_percentage=0.8, seed=1)

model.train(train_X, train_Y)

predictions = model.predict(test_X)
print(predictions)
