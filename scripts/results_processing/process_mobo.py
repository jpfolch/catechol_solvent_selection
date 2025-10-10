import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catechol.data.bo_benchmark import BOBenchmark
import scienceplots

plt.style.use(["science", "no-latex", "grid"])

bench = BOBenchmark(featurization="spange_descriptors")

features = ["spange_descriptors"]
strategies = ["random_scalars", "random"]
models = []

clean_results = pd.DataFrame()

gds_final = {
    "random_scalars": [],
    "random": []
}
igds_final = {
    "random_scalars": [],
    "random": []
}
mpfes_final = {
    "random_scalars": [],
    "random": []
}

for feature in features:
    for strategy in strategies:
        model_name = f"GPModel-indep-{feature}/{strategy}"

        gd = []
        igd = []
        mpfe = []

        for i in range(24):
            results = pd.read_csv(f"results/mobo/{model_name}/{i}.csv")
            gd.append(results["GD"])
            igd.append(results["IGD"])
            mpfe.append(results["MPFE"])

        gds = np.array(gd)
        igds = np.array(igd)
        mpfes = np.array(mpfe)

        gds_mean = np.mean(gds, axis=0)
        gds_final[strategy].append(gds_mean[[0, 24, 49, 74, 98]].tolist())

        igds_mean = np.mean(igds, axis=0)
        igds_final[strategy].append(igds_mean[[0, 24, 49, 74, 98]].tolist())

        mpfes_mean = np.mean(mpfes, axis=0)
        mpfes_final[strategy].append(mpfes_mean[[0, 24, 49, 74, 98]].tolist())

# build a table
gds_table = pd.DataFrame()
igds_table = pd.DataFrame()
mpfes_table = pd.DataFrame()

gds_table["Iteration"] = ["1", "25", "50", "75", "100"]
igds_table["Iteration"] = ["1", "25", "50", "75", "100"]
mpfes_table["Iteration"] = ["1", "25", "50", "75", "100"]

gds_table["Avg GD (Random)"] = [gds_final["random"][0][i] for i in range(5)]
gds_table["Avg GD (MOBO)"] = [gds_final["random_scalars"][0][i] for i in range(5)]

igds_table["Avg IGD (Random)"] = [igds_final["random"][0][i] for i in range(5)]
igds_table["Avg IGD (MOBO)"] = [igds_final["random_scalars"][0][i] for i in range(5)]

mpfes_table["Avg MPFE (Random)"] = [mpfes_final["random"][0][i] for i in range(5)]
mpfes_table["Avg MPFE (MOBO)"] = [mpfes_final["random_scalars"][0][i] for i in range(5)]

# print
print("GD Table:")
print(gds_table.to_string(index=False))
print("\nIGD Table:")
print(igds_table.to_string(index=False))
print("\nMPFE Table:")
print(mpfes_table.to_string(index=False))