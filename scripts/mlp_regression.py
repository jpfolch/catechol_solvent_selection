import os
os.chdir('./src')

from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT, TARGET_LABELS
os.chdir('..')

from catechol.models import MLPModel  # adjust import path if needed
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
from catechol import metrics
import matplotlib.pyplot as plt

# --- Optional: Load featurization dictionary if needed ---
use_solvent_embedding = False
featurization_lookup = None
if use_solvent_embedding:
    from catechol.data.featurizations import load_featurization_lookup
    featurization_lookup = load_featurization_lookup()

# --- Initialize MLP model ---
model = MLPModel(
    learning_rate = 1e-3,
    dropout = 0.1,
    epochs = 100,
    use_validation = "leave_one_solvent_out",
    batch_size=32,
    use_solvent_embedding=use_solvent_embedding,
    featurization_lookup=featurization_lookup,
    solvent_column="SOLVENT NAME"
)

# --- Load dataset ---
X, Y = load_single_solvent_data()
X = X[['Residence Time', 'Temperature', 'SOLVENT NAME']]  # drop SMILES, since not used in MLP

# --- Train model ---
model.train(train_X = X, train_Y = Y)

# --- Evaluate on a split ---
split_generator = generate_leave_one_out_splits(X, Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
model.train(train_X=train_X, train_Y=train_Y)

# --- Predict and calculate metrics ---
train_X, test_X = train_test_split(X, train_percentage=0.8, seed=1)
train_Y, test_Y = train_test_split(Y, train_percentage=0.8, seed=1)

predictions = model.predict(test_X)
print(predictions)

mse = metrics.mse(predictions, test_Y)
nlpd = metrics.nlpd(predictions, test_Y)
print(f"{mse=}, {nlpd=}")

# --- Plotting ---
plot_solvent_prediction(model, test_X, test_Y)

# Optionally visualize more splits
for _ in range(2):
    (train_X, train_Y), (test_X, test_Y) = next(split_generator)
    plot_solvent_prediction(model, test_X, test_Y)

plt.show()
