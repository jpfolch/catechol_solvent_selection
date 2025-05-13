
import matplotlib.pyplot as plt
import scienceplots
import os

from catechol.data.loader import load_single_solvent_data

plt.style.use(["science", "no-latex", "grid"])

X, Y = load_single_solvent_data()

# get all solvents
solvents = X["SOLVENT NAME"].unique()

directory = "figures/single_solvent_plots"
# create a directory to save the plots
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(directory + '/225_degrees'):
    os.makedirs(directory + '/225_degrees')

if not os.path.exists(directory + '/175_degrees'):
    os.makedirs(directory + '/175_degrees')

if not os.path.exists(directory + '/200_degrees'):
    os.makedirs(directory + '/200_degrees')


for solvent in solvents:
    # filter the data for the solvent
    mask = X["SOLVENT NAME"] == solvent
    X_solvent = X[mask]
    Y_solvent = Y[mask]
    # now filter for higher temperature
    mask = X_solvent["Temperature"] > 224
    X_solvent = X_solvent[mask]
    Y_solvent = Y_solvent[mask]

    # if there are more than 0 rows, plot the data
    if len(X_solvent) > 0:
        # create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # plot the data
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["SM"] * 100, label="Starting Material"
        )
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["Product 2"] * 100, label="Product 2"
        )
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["Product 3"] * 100, label="Product 3"
        )

        # set the title and labels
        ax.set_xlabel("Residence Time / min", fontsize=16)
        ax.set_ylabel("Yield / %", fontsize=16)

        # set tick sizes
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)

        # title
        ax.set_title(f"{solvent} 225$\degree$C", fontsize=16)

        # limits
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 101)

        # add a legend
        ax.legend(fontsize=16)

        # save the plot
        plt.savefig(
            os.path.join(directory, "225_degrees", f"{solvent}_225.png"),
            bbox_inches="tight",
            dpi=300,
        )

        # show the plot
        plt.show()


# repeat for low temperature
for solvent in solvents:
    # filter the data for the solvent
    mask = X["SOLVENT NAME"] == solvent
    X_solvent = X[mask]
    Y_solvent = Y[mask]
    # now filter for lower temperature
    mask = X_solvent["Temperature"] < 176
    X_solvent = X_solvent[mask]
    Y_solvent = Y_solvent[mask]

    if len(X_solvent) > 0:
        # create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # plot the data
        ax.scatter(X_solvent["Residence Time"], Y_solvent["SM"] * 100, label="Starting Material")
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["Product 2"] * 100, label="Product 2"
        )
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["Product 3"] * 100, label="Product 3"
        )

        # set the title and labels
        ax.set_xlabel("Residence Time / min", fontsize=16)
        ax.set_ylabel("Yield / %", fontsize=16)

        # set tick sizes
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)

        # title
        ax.set_title(f"{solvent} 175$\degree$C", fontsize=16)

        # limits
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 101)

        # add a legend
        ax.legend(fontsize=16)

        # save the plot
        plt.savefig(
            os.path.join(directory, "175_degrees", f"{solvent}_175.png"),
            bbox_inches="tight",
            dpi=300,
        )

        # show the plot
        plt.show()

# now do the green solvents at 200
for solvent in solvents:
    # filter the data for the solvent
    mask = X["SOLVENT NAME"] == solvent
    X_solvent = X[mask]
    Y_solvent = Y[mask]
    # now filter for lower temperature
    mask = (199 <= X_solvent["Temperature"]) & (X_solvent["Temperature"] <= 200)
    X_solvent = X_solvent[mask]
    Y_solvent = Y_solvent[mask]

    if len(X_solvent) > 0:
        # create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # plot the data
        ax.scatter(X_solvent["Residence Time"], Y_solvent["SM"] * 100, label="Starting Material")
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["Product 2"] * 100, label="Product 2"
        )
        ax.scatter(
            X_solvent["Residence Time"], Y_solvent["Product 3"] * 100, label="Product 3"
        )

        # set the title and labels
        ax.set_xlabel("Residence Time / min", fontsize=16)
        ax.set_ylabel("Yield / %", fontsize=16)

        # set tick sizes
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)

        # title
        ax.set_title(f"{solvent} 200$\degree$C", fontsize=16)

        # add a legend
        ax.legend(fontsize=16)

        # limits
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 101)

        # save the plot
        plt.savefig(
            os.path.join(directory, "200_degrees", f"{solvent}_200.png"),
            bbox_inches="tight",
            dpi=300,
        )

        # show the plot
        plt.show()
