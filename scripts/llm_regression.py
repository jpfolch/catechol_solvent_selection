import os

import pandas as pd

os.chdir("./src")
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)

os.chdir("..")

from catechol import metrics
from catechol.models import LLMModel

# --- Parameter grids ---
model_names = [
    "seyonec/ChemBERTa-zinc-base-v1",
    "rxnfp-pretrained",
]
# dropout_head_values = [1e-6, 1e-4]
# dropout_backbone_values = [1e-6, 1e-4]
# lr_head_values = [1e-4, 1e-3]
# lr_backbone_values = [1e-5, 1e-4]
dropout_values = [0.1, 0.3, 0.5, 0.75]
lr_values = [1e-5, 1e-4, 1e-3]
pooler_output_values = [True, False]
fixed_backbone_values = [True, False]
# --- Load dataset ---
X, Y = load_single_solvent_data()
X = X[["Residence Time", "Temperature", "Reaction SMILES", "SOLVENT NAME"]]

results = []

# --- Loop over all parameter combinations ---
for model_name in model_names:
    for dr in dropout_values:
        for lr in lr_values:
            for fixed_backbone in fixed_backbone_values:
                for pooler_output in pooler_output_values:
                    print(
                        f"\nTraining model={model_name}, lr={lr}, dr={dr}, pooler_output={pooler_output}"
                    )

                    split_generator = generate_leave_one_out_splits(X, Y)
                    mse_scores = []
                    solvent_names = []

                    for split_idx, ((train_X, train_Y), (test_X, test_Y)) in enumerate(
                        split_generator, 1
                    ):
                        # Initialize model
                        model = LLMModel(
                            model_name=model_name,
                            freeze_backbone=fixed_backbone,
                            learning_rate_backbone=lr,
                            learning_rate_head=lr,
                            dropout_backbone=dr,
                            dropout_head=dr,
                            use_pooler_output=pooler_output,
                            custom_head=None,
                            max_length_padding=None,
                            epochs=200,
                            time_limit=10800,
                            # use_validation="leave_one_solvent_out",
                            batch_size=32,
                        )

                        # Train and evaluate
                        model.train(train_X=train_X, train_Y=train_Y)
                        test_X, test_Y = replace_repeated_measurements_with_average(
                            test_X, test_Y
                        )
                        predictions = model.predict(test_X)
                        mse = metrics.mse(predictions, test_Y)

                        solvent = test_X["SOLVENT NAME"].unique()[0]
                        mse_scores.append(mse)
                        solvent_names.append(solvent)

                        print(f"  Split {split_idx} ({solvent}): MSE = {mse:.4f}")

                    avg_mse = sum(mse_scores) / len(mse_scores)

                    results.append(
                        {
                            "model_name": model_name,
                            "dropout_rate": dr,
                            "learning_rate": lr,
                            "pooler_output": pooler_output,
                            "avg_mse": avg_mse,
                            "all_mse": mse_scores,
                            "solvent_names": solvent_names,
                        }
                    )

# --- Convert results to expanded DataFrame ---
results_expanded = []
for entry in results:
    base = {
        "model_name": entry["model_name"],
        "dropout_head": entry["dropout_head"],
        "dropout_backbone": entry["dropout_backbone"],
        "lr_head": entry["lr_head"],
        "lr_backbone": entry["lr_backbone"],
        "avg_mse": entry["avg_mse"],
    }
    for mse, solvent in zip(entry["all_mse"], entry["solvent_names"]):
        base[f"mse_{solvent}"] = mse
    results_expanded.append(base)

results_df = pd.DataFrame(results_expanded)
results_df.sort_values(by="avg_mse", inplace=True)
results_df.to_csv("llm_mse_hyperparam_results.csv", index=False)

print("\nTop configurations by MSE:")
print(results_df.head())
