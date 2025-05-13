import os

import matplotlib.pyplot as plt
import scienceplots

from catechol.data.loader import load_solvent_ramp_data

plt.style.use(["science", "no-latex", "grid"])

directory = "figures/full_surfaces"
# create a directory to save the plots
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(directory + '/SM'):
    os.makedirs(directory + '/SM')

if not os.path.exists(directory + '/Product 2'):
    os.makedirs(directory + '/Product 2')

if not os.path.exists(directory + '/Product 3'):
    os.makedirs(directory + '/Product 3')

# load the full data
X, Y = load_solvent_ramp_data()

# plot for each experiment
exp_nums = X["EXP NUM"].unique()

vmax = {
    "SM": 100,
    "Product 2": 50,
    "Product 3": 50,
}

for exp in exp_nums:
    # get the variables
    X_exp = X[X["EXP NUM"] == exp]
    Y_exp = Y[X["EXP NUM"] == exp]

    x = X_exp["Residence Time"]
    y = X_exp["SolventB%"] * 100
    z = X_exp["Temperature"]

    # plot each 3d surface
    for prod in ["SM", "Product 2", "Product 3"]:

        # create color map for values between 0 and 100, to represent yields
        color_map = plt.get_cmap("viridis")
        # for values between 0 and 100
        norm = plt.Normalize(vmin=0, vmax=vmax[prod])

        c = Y_exp[prod] * 100
        # map the color with the correct normalization
        c = color_map(norm(c))

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
        # scatter
        ax.scatter(x, y, z, c=c, cmap=color_map, marker="o", edgecolor="k", s=100, depthshade=False)

        ax.set_xlabel("Residence Time / min", fontsize=16, labelpad=10)
        ax.set_ylabel("SolventB%", fontsize=16, labelpad=10)
        ax.set_zlabel("Temperature / $\degree$C", fontsize=16, labelpad=10)

        # set the title
        solvent_a = X_exp["SOLVENT A NAME"].iloc[0]
        solvent_b = X_exp["SOLVENT B NAME"].iloc[0]

        title = f"{solvent_a} to {solvent_b}"
        # if title is too long, split it into two lines based on solvent
        if len(title) > 30:
            title = title.split(" to ")
            title = f"{title[0]} to \n{title[1]}"
        
        ax.set_title(title, fontsize=16, pad=-30, y=1.05)

        # set the view angle
        ax.view_init(30, 45)

        # set the limits
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 100)
        ax.set_zlim(150, 250)

        # add the colorbar
        mappable = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        mappable.set_array(c)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
        if prod == "SM":
            prod_name = "Starting Material"
        else:
            prod_name = prod
        cbar.set_label(f"{prod_name} Yield / %", fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        # increase colorbar padding
        # cbar.ax.yaxis.set_label_coords(1.05, 0.5)
        # make the colorbar smaller

        # set tick sizes
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)
        # set the font size
        for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]):
            item.set_fontsize(16)
        # set the font size for the ticks
        for item in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(16)

        ax.set_box_aspect(None, zoom=0.75)
        # save the figure
        fig.savefig(f"{directory}/{prod}/{exp}.png", dpi=300, bbox_inches="tight")

        # show
        plt.show()