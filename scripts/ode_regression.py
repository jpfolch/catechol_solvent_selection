import os
import pandas as pd

os.chdir("./src")
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)

os.chdir("..")

from catechol import models
from catechol import metrics

# --- Parameter grids ---
# TODO: Adjust the learning rate 
model_lists = ["LODEModel"] # "LODEModel", "EODEModel", 
cfg_dicts = {
    "NODEModel": {"device": "cuda", "state_dim": 3, "h_dim_ode": 64},
    "EODEModel": {
        "device": "cuda",
        "state_dim": 3,
        "dynamics_dim": 32,
        "h_dim_ode": 64,
        "h_dim_dynmcs": 64,
        "h_dim_dec": 64,
    },
    "LODEModel": {"device": "cuda", "state_dim": 3, "h_dim_ode": 64},
}
train_epochs = {"LODEModel": 100, "EODEModel": 100, "NODEModel": 100}
learning_rate_dicts = {"LODEModel": 1e-3, "EODEModel": 1e-3, "NODEModel": 1e-3}
featurization_type_values = [
    "acs_pca_descriptors"
]  # ['drfps', 'fragprints'] # ['acs_pca_descriptors', ]
featurization_dims = {"acs_pca_descriptors": 5, "drfps": 2048, "fragprints": None}
# --- Load dataset ---
X, Y = load_single_solvent_data()
X = X[["Residence Time", "Temperature", "SOLVENT NAME"]]

use_validation = None
results = []

for _ODEModel in model_lists:
    for featurization_type in featurization_type_values:

        split_generator = generate_leave_one_out_splits(X, Y)
        mse_scores = []
        solvent_test = []
        split_index = 0
        for (train_X, train_Y), (test_X, test_Y) in split_generator:
            # remove all time > 15 to remove duplication
            mask = train_X["Residence Time"] <= 15
            train_X = train_X[mask]
            train_Y = train_Y[mask]

            split_index += 1
            # Update featureization in cfgs

            # Initialize model
            Model = getattr(models, _ODEModel)(
                **cfg_dicts[_ODEModel],
                featurization=featurization_type,
                featurization_dim=featurization_dims[featurization_type],
            )

            # Train and evaluate
            Model.train(
                train_X=train_X,
                train_Y=train_Y,
                learning_rate=learning_rate_dicts[_ODEModel],
                train_epoch=train_epochs[_ODEModel],
            )
            test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
            predictions = Model.predict(test_X)
            mse = metrics.mse(predictions, test_Y)
            mse_scores.append(mse)
            solvent_test.append(test_X.loc[:, "SOLVENT NAME"].unique())
            print(f"  Split {split_index}: MSE = {mse:.4f}")

        avg_mse = sum(mse_scores) / len(mse_scores)

        results.append(
            {   
                "model": _ODEModel,
                "learning_rate": learning_rate_dicts[_ODEModel],
                "featurization": featurization_type,
                "avg_mse": avg_mse,
                "all_mse": mse_scores,
                "solvent_test": solvent_test,
            }
        )

# Convert to DataFrame for easier viewing/sorting
# Expand 'all_mse' list into separate columns
results_expanded = []
for entry in results:
    base = {
        "model": entry["model"],  # add this line
        "learning_rate": entry["learning_rate"],
        "featurization": entry["featurization"],
        "avg_mse": entry["avg_mse"],
    }

    for val1, val2 in zip(entry["all_mse"], entry["solvent_test"]):
        base[f"mse_split_{val2}"] = val1

    results_expanded.append(base)

results_df = pd.DataFrame(results_expanded)
results_df.sort_values(by="avg_mse", ascending=True, inplace=True)
# results_df.to_csv("mlp_mse_hyperparam_results.csv", index=False)

print("\nTop configurations by MSE:")
print(results_df.head())
results_df.to_csv("full_results.csv")
