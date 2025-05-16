import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catechol.data.bo_benchmark import BOBenchmark

plt.style.use(["science", "no-latex", "grid"])

bench = BOBenchmark(featurization="spange_descriptors")

features = ["spange_descriptors"]
strategies = ["random", "ucb", "ei"]
models = []

clean_results = pd.DataFrame()

for feature in features:
    for strategy in strategies:
        model_name = f"GPModel-indep-{feature}/{strategy}"

        best_observations = []

        for i in range(29):
            results = pd.read_csv(f"results/bo/{model_name}/{i}.csv")
            best_observations.append(results["Best value"])

        mses = np.array(best_observations)
        mses_10_quantile = np.quantile(mses, 0.1, axis=0)
        mses_90_quantile = np.quantile(mses, 0.9, axis=0)
        mses_50_quantile = np.quantile(mses, 0.5, axis=0)

        clean_results[f"{model_name}-mse"] = mses_50_quantile
        clean_results[f"{model_name}-mse_10"] = mses_10_quantile
        clean_results[f"{model_name}-mse_90"] = mses_90_quantile

        models.append(model_name)

# now drop rows with nans
clean_results = clean_results.dropna()
clean_results = clean_results.reset_index(drop=True)

# plot
fig, ax = plt.subplots(figsize=(6, 4))

for model in models:
    if "acs" in model:
        continue

    mean_col = f"{model}-mse"

    if "random" in model:
        label = "Random"
    elif "ei" in model:
        label = "Expected Improvement"
    elif "ucb" in model:
        label = "Upper Confidence Bound"

    ax.plot(
        clean_results.index,
        clean_results[mean_col],
        label=label,
    )
    ax.fill_between(
        clean_results.index,
        clean_results[mean_col + "_10"],
        clean_results[mean_col + "_90"],
        alpha=0.2,
    )

ax.set_xlabel("Iteration", fontsize=16)
ax.set_ylabel("Best value", fontsize=16)

ax.set_ylim(1.0, 3.1)
ax.set_xlim(0, 50)

optimum = bench.get_optimum()[1]
ax.axhline(
    y=optimum,
    color="black",
    linestyle="--",
)

# set tick sizes
ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="both", which="minor", labelsize=16)

ax.legend(fontsize=16)

# save the figure
fig.savefig("figures/bo.pdf", dpi=300, bbox_inches="tight")

plt.show()
