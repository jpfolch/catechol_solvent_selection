import matplotlib.pyplot as plt
from catechol.data.data_labels import INPUT_LABELS_FULL_DATA
from catechol.data.loader import (
    generate_leave_one_ramp_out_splits,
    load_solvent_ramp_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import GPModel
from catechol.plots.plot_solvent_prediction import plot_solvent_ramp_prediction
import matplotlib.lines as mlines
from catechol.plots import style
import scienceplots
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams["font.size"] = 12


# model = GPModel(featurization="spange_descriptors")
model = GPModel(
    multitask=False, use_input_warp=False, featurization="spange_descriptors"
)
X, Y = load_solvent_ramp_data()
# remove unnecessary columns
X = X[INPUT_LABELS_FULL_DATA]

split_generator = generate_leave_one_ramp_out_splits(X, Y)

# skip 5 times for 2-Methyltetrahydrofuran [2-MeTHF]-Diethyl Ether [Ether]
# skip 7 times for "DMA [N,N-Dimethylacetamide]-Decanol"
for _ in range(5):
    next(split_generator)

(train_X, train_Y), (test_X, test_Y) = next(split_generator)

model.train(train_X, train_Y)

test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)

fig, axs = plot_solvent_ramp_prediction(model, test_X, test_Y)

fig.set_size_inches((8, 3))

# clean up the legend
axs[0].legend().set_visible(False)
axs[1].legend().set_visible(False)

legend_lines = [
    mlines.Line2D([], [], color=clr, label=style.TARGET_TO_LABEL[name], marker="o", markeredgecolor="black", linestyle="None") for name, clr in style.TARGET_TO_COLOR.items()
] 
fig.legend(handles=legend_lines, loc="outside lower center", ncol=4,bbox_to_anchor=(0.5, -0.15),
    frameon=True,
)
fig.savefig("figures/solvent_ramp_prediction.pdf", bbox_inches="tight")

for _ in range(5):
    (train_X, train_Y), (test_X, test_Y) = next(split_generator)

plot_solvent_ramp_prediction(model, test_X, test_Y)
plt.show()
