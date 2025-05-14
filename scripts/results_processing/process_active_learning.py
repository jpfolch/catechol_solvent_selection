import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os

plt.style.use(["science", "no-latex", "grid"])

features = ['acs_pca_descriptors', 'spange_descriptors']
strategies = ['random', 'entropy', 'mutual_information']
models = []

clean_results = pd.DataFrame()

for feature in features:
    for strategy in strategies:
        model_name = f"GPModel-indep-{feature}/{strategy}"

        mses = []
        nlpds = []

        for i in range(29):
            results = pd.read_csv(f"results/active_learning/{model_name}/{i}.csv")
            mses.append(results['mse'])
            nlpds.append(results['nlpd'])

        mses = np.array(mses)
        mses_25_quantile = np.quantile(mses, 0.25, axis=0)
        mses_75_quantile = np.quantile(mses, 0.75, axis=0)
        nlpds = np.array(nlpds)

        clean_results[f'{model_name}-mse'] = mses.mean(axis=0)
        clean_results[f'{model_name}-mse_25'] = mses_25_quantile
        clean_results[f'{model_name}-mse_75'] = mses_75_quantile
        clean_results[f'{model_name}-nlpd'] = nlpds.mean(axis=0)

        models.append(model_name)

# now drop rows with nans
clean_results = clean_results.dropna()
clean_results = clean_results.reset_index(drop=True)

# plot
fig, ax = plt.subplots(figsize=(6, 4))

for model in models:
    if 'acs' in model:
        continue

    mean_col = f'{model}-mse'

    if 'random' in model:
        label = 'Random'
    elif 'entropy' in model:
        label = 'Entropy'
    elif 'mutual_information' in model:
        label = 'Mutual Information'

    ax.plot(
        clean_results.index,
        clean_results[mean_col],
        label=label,
    )
    ax.fill_between(
        clean_results.index,
        clean_results[mean_col + '_25'],
        clean_results[mean_col + '_75'],
        alpha=0.2,
    )

ax.set_xlabel("Iteration", fontsize=16)
ax.set_ylabel("MSE", fontsize=16)

ax.set_ylim(0, 0.04)

# set tick sizes
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)

ax.legend(fontsize=16)

# save the figure
fig.savefig("figures/active_learning.pdf", dpi=300, bbox_inches="tight")

plt.show()