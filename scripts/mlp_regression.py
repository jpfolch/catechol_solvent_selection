import os
import pandas as pd

os.chdir('./src')
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)
os.chdir('..')

from catechol.models import MLPModel
from catechol import metrics

# --- Parameter grids ---
dropout_values = [0.0, 0.1, 0.3, 0.5]
learning_rates = [1e-5, 1e-4, 1e-3]

# --- Load dataset ---
X, Y = load_single_solvent_data()
X = X[['Residence Time', 'Temperature', 'SOLVENT NAME']]

use_validation = None
results = []

for dropout in dropout_values:
    for lr in learning_rates:
        print(f"\nTraining with dropout={dropout}, learning_rate={lr}")
        
        split_generator = generate_leave_one_out_splits(X, Y)
        mse_scores = []
        solvent_test = []
        split_index = 0
        for (train_X, train_Y), (test_X, test_Y) in split_generator:
            split_index += 1

            # Initialize model
            model = MLPModel(
                learning_rate=lr,
                dropout=dropout,
                epochs=200,
                #use_validation=use_validation,
                batch_size=32,
                featurization_type="acs_pca_descriptors"
            )

            # Train and evaluate
            model.train(train_X=train_X, train_Y=train_Y)
            test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
            predictions = model.predict(test_X)
            mse = metrics.mse(predictions, test_Y)
            mse_scores.append(mse)
            solvent_test.append(test_X.loc[:,'SOLVENT NAME'].unique())
            print(f"  Split {split_index}: MSE = {mse:.4f}")

        avg_mse = sum(mse_scores) / len(mse_scores)

        results.append({
            'dropout': dropout,
            'learning_rate': lr,
            'avg_mse': avg_mse,
            'all_mse': mse_scores,
            'solvent_test': solvent_test
        })

# Convert to DataFrame for easier viewing/sorting
# Expand 'all_mse' list into separate columns
results_expanded = []
for entry in results:
    base = {
        'dropout': entry['dropout'],
        'learning_rate': entry['learning_rate'],
        'avg_mse': entry['avg_mse'],
    }
    for val1, val2 in zip(entry['all_mse'], entry['solvent_test']):
        base[f'mse_split_{val2}'] = val1
        
    results_expanded.append(base)

results_df = pd.DataFrame(results_expanded)
results_df.sort_values(by='avg_mse', ascending=True, inplace=True)
#results_df.to_csv("mlp_mse_hyperparam_results.csv", index=False)

print("\nTop configurations by MSE:")
print(results_df.head())
