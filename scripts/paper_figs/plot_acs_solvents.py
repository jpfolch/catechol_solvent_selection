import os

import matplotlib.pyplot as plt
from catechol.data.featurizations import featurize_input_df
from catechol.data.loader import load_solvent_ramp_data

directory = "figures/acs_solvent_ramps_coloured"
# create a directory to save the plots
if not os.path.exists(directory):
    os.makedirs(directory)

target = "Product 3"

# load the full data
X, Y = load_solvent_ramp_data()

# featurize the data
X = featurize_input_df(X, featurization="acs_pca_descriptors")

# choose data only at specific temperatures and residence times
mask = X["Temperature"] < 176
mask &= X["Residence Time"] > 14

X = X[mask]
Y = Y[mask]

x1 = X["A_PC1"] * (1 - X["SolventB%"]) + X["B_PC1"] * X["SolventB%"]
x2 = X["A_PC2"] * (1 - X["SolventB%"]) + X["B_PC2"] * X["SolventB%"]

# plot with colour based on the yield of SM
fig, ax = plt.subplots(figsize=(8, 6))

# plot the data
ax.scatter(
    x1,
    x2,
    c=Y[target] * 100,
    cmap="viridis",
    marker="o",
    edgecolor="k",
    s=100,
)

# find the single solvent points
single_solvent_mask = X["SolventB%"] == 0
ax.scatter(
    x1[single_solvent_mask],
    x2[single_solvent_mask],
    c=Y[target][single_solvent_mask] * 100,
    marker="*",
    edgecolor="k",
    s=500,
)

# set the title and labels
ax.set_title(f"{target} Yield", fontsize=16)
ax.set_xlabel("PC1", fontsize=16)
ax.set_ylabel("PC2", fontsize=16)
# ax.set_xlim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)
# set tick sizes
ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="both", which="minor", labelsize=16)
# set the color bar
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label(f"{target} Yield", fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.show()
