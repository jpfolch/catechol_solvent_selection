# Data collected from training on full data, with the test solvent
# 1,1,1,3,3,3-Hexafluoropropan-2-ol-2-Methyltetrahydrofuran [2-MeTHF]

import matplotlib.pyplot as plt
import numpy as np

import scienceplots
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams["font.size"] = 12

def kumaraswamy_cdf(x, a, b):
    return 1 - (1 - x**a) ** b

warp_params_time = (0.6614, 1.6063)
warp_params_solvent = (0.6133, 0.7937)

fig, axs = plt.subplots(ncols=2, figsize=(8, 3), subplot_kw={"aspect": "equal"})
axs: list[plt.Axes]
fig.subplots_adjust(wspace=1.0)
x = np.linspace(0, 1, 100)

axs[0].plot(x, kumaraswamy_cdf(x, *warp_params_time), linewidth=3)
axs[1].plot(x, kumaraswamy_cdf(x, *warp_params_solvent), linewidth=3)

for ax in axs:
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

# if you want the residence time in minutes, uncomment below
# axs[0].set_xticks([0.0, 0.5, 1.0], ["0.0", "7.5", "15.0"])
# axs[0].set_yticks([0.0, 0.5, 1.0], ["0.0", "7.5", "15.0"])

axs[0].set_xlabel("Normalized residence time")
axs[0].set_ylabel("Warped residence time")

axs[1].set_xlabel("Solvent B%")
axs[1].set_ylabel("Warped Solvent B%")

fig.savefig("figures/input_warping.pdf", bbox_inches="tight")
plt.show()