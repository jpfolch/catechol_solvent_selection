import os

import matplotlib.pyplot as plt
import scienceplots

from catechol.data.loader import load_solvent_ramp_data

plt.style.use(["science", "no-latex", "grid"])

directory = "figures/solvent_ramps"
# create a directory to save the plots
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(directory + '/225_degrees'):
    os.makedirs(directory + '/225_degrees')

if not os.path.exists(directory + '/175_degrees'):
    os.makedirs(directory + '/175_degrees')

# load the full data
X, Y = load_solvent_ramp_data()

# plot for each experiment
exp_nums = X["EXP NUM"].unique()

for exp in exp_nums:

    X_exp = X[X["EXP NUM"] == exp]
    Y_exp = Y[X["EXP NUM"] == exp]

    # now filter for higher temperature
    mask = X_exp["Temperature"] > 224
    X_exp = X_exp[mask]
    Y_exp = Y_exp[mask]

    # filter for higher residence time
    mask = X_exp["Residence Time"] > 14
    X_exp = X_exp[mask]
    Y_exp = Y_exp[mask]

    # if there are more than 0 rows, plot the data
    if len(X_exp) > 0:
        # plot solvent B percentage against SM and products
        fig, ax = plt.subplots(figsize=(8, 6))
        # plot the data
        ax.scatter(X_exp["SolventB%"] * 100, Y_exp["SM"] * 100, label="Starting Material")
        ax.scatter(X_exp["SolventB%"] * 100, Y_exp["Product 2"] * 100, label="Product 2")
        ax.scatter(X_exp["SolventB%"] * 100, Y_exp["Product 3"] * 100, label="Product 3")

        # set the title and labels
        solvent_a = X_exp["SOLVENT A NAME"].iloc[0]
        solvent_b = X_exp["SOLVENT B NAME"].iloc[0]

        title = f"{solvent_a} to {solvent_b}"
        # if title is too long, split it into two lines based on solvent
        if len(title) > 30:
            title = title.split(" to ")
            title = f"225$\degree$C: {title[0]} to \n{title[1]}"

        ax.set_title(title, fontsize=16)

        ax.set_xlabel("SolventB% / %", fontsize=16)
        ax.set_ylabel("Yield / %", fontsize=16)

        # set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

        # limits
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 101)

        # legend
        ax.legend(fontsize=16)

        # save the figure
        fig.savefig(f"{directory}/225_degrees/{solvent_a}_{solvent_b}_exp_{exp}.png", dpi=300, bbox_inches="tight")

        # show the plot
        plt.show()

# now plot for 175 degrees

for exp in exp_nums:

    X_exp = X[X["EXP NUM"] == exp]
    Y_exp = Y[X["EXP NUM"] == exp]

    # now filter for higher temperature
    mask = X_exp["Temperature"] < 176
    X_exp = X_exp[mask]
    Y_exp = Y_exp[mask]

    # filter for higher residence time
    mask = X_exp["Residence Time"] > 14
    X_exp = X_exp[mask]
    Y_exp = Y_exp[mask]

    # if there are more than 0 rows, plot the data
    if len(X_exp) > 0:
        # plot solvent B percentage against SM and products
        fig, ax = plt.subplots(figsize=(8, 6))
        # plot the data
        ax.scatter(X_exp["SolventB%"] * 100, Y_exp["SM"] * 100, label="Starting Material")
        ax.scatter(X_exp["SolventB%"] * 100, Y_exp["Product 2"] * 100, label="Product 2")
        ax.scatter(X_exp["SolventB%"] * 100, Y_exp["Product 3"] * 100, label="Product 3")

        # set the title and labels
        solvent_a = X_exp["SOLVENT A NAME"].iloc[0]
        solvent_b = X_exp["SOLVENT B NAME"].iloc[0]

        title = f"{solvent_a} to {solvent_b}"
        # if title is too long, split it into two lines based on solvent
        if len(title) > 30:
            title = title.split(" to ")
            title = f"175$\degree$C: {title[0]} to \n{title[1]}"

        ax.set_title(title, fontsize=16)

        ax.set_xlabel("SolventB%", fontsize=16)
        ax.set_ylabel("Yield / %", fontsize=16)

        # set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

        # limits
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 101)

        # legend
        ax.legend(fontsize=16)

        # save the figure
        fig.savefig(f"{directory}/175_degrees/{solvent_a}_{solvent_b}_exp_{exp}.png", dpi=300, bbox_inches="tight")

        # show the plot
        plt.show()