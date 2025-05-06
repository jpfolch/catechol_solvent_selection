import os
os.chdir('./src')

from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
)

os.chdir('..')

from catechol.models import MLPModel  # adjust import path if needed
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
from catechol import metrics
import matplotlib.pyplot as plt

use_validation = "leave_one_solvent_out"
# --- Initialize MLP model ---
model = MLPModel(
    learning_rate = 1e-5,
    dropout = 0.1,
    epochs = 5000,
    use_validation = use_validation,
    batch_size=32,
    featurization_type = "acs_pca_descriptors"
)

# --- Load dataset ---
X, Y = load_single_solvent_data()
X = X[['Residence Time', 'Temperature', 'SOLVENT NAME']]  # drop SMILES, since not used in MLP

# --- Train model ---
model.train(train_X = X, train_Y = Y)
epochs = list(range(len(model.train_losses)))
plt.plot(epochs, model.train_losses, label='Training Loss')
if use_validation is not None:
    plt.plot(epochs, model.val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
# Optionally visualize more splits
split_generator = generate_leave_one_out_splits(X, Y)
for _ in range(2):
    (train_X, train_Y), (test_X, test_Y) = next(split_generator)
    plot_solvent_prediction(model, test_X, test_Y)
    predictions = model.predict(test_X)
    mse = metrics.mse(predictions, test_Y)
    nlpd = metrics.nlpd(predictions, test_Y)
    print(f"{mse=}, {nlpd=}")

plt.show()
