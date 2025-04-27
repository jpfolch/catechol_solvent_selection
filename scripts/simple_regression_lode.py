from catechol.data.loader import load_single_solvent_data, train_test_split
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.models import LantentODE
from catechol import metrics
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)
from matplotlib import pyplot as plt

model = LantentODE(
    latent_state_dim=16,
    latent_dynamics_dim=16,
    h_dim_ode=32,
    h_dim_x0=32,
    h_dim_dynmcs=32,
    h_dim_dec=32,
    featurization_dim=5,
    featurization="acs_pca_descriptors",
    device="cuda",
)
X, Y = load_single_solvent_data()
# remove unnecessary columns
X = X[INPUT_LABELS_SINGLE_SOLVENT]

train_X, train_Y = load_single_solvent_data()
# remove unnecessary columns
train_X = train_X[INPUT_LABELS_SINGLE_SOLVENT]

# Filter where Residence TIME <= 15 as the termination measurement is pretty noisy
mask = train_X["Residence Time"] <= 15

train_X = train_X[mask]
train_Y = train_Y[mask]

train_X, test_X = train_test_split(train_X, train_percentage=0.8, seed=1)
train_Y, test_Y = train_test_split(train_Y, train_percentage=0.8, seed=1)

model.train(
    train_X,
    train_Y,
    learning_rate=1e-3,
    train_epoch=100,
    use_pretrained_model=False,
    train_dir="/homes/jqing/codes/catechol_solvent_selection/exps/lode/",
    save_freq=10,
    mc_sample_num=32,
    kl_weight=1.0,
    validation_fraction=0.1,
)


predictions = model.predict(test_X)
print(predictions)

# calculate some metrics
mse = metrics.mse(predictions, test_Y)
nlpd = metrics.nlpd(predictions, test_Y)
print(f"test: {mse=}, {nlpd=}")

# plot the predictions
split_generator = generate_leave_one_out_splits(X, Y)
plot_solvent_prediction(model, test_X, test_Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)

plt.savefig("LODE_pred.png", dpi=300)
