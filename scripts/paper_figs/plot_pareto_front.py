import matplotlib.pyplot as plt
from catechol.data.loader import load_solvent_ramp_data
import scienceplots
plt.style.use(["science", "no-latex", "grid"])
X, Y = load_solvent_ramp_data()

yields_sm = Y["SM"] * 100
yields_product_2 = Y["Product 2"] * 100
yields_product_3 = Y["Product 3"] * 100

selectivity = yields_product_2 / (yields_product_2 + yields_product_3) * 100
product_yield = yields_product_2 + yields_product_3

# Create a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(product_yield, selectivity, edgecolor='k', s=15)
plt.xlabel("Product Yield / %", fontsize=16)
plt.ylabel("Selectivity / %", fontsize=16)
plt.title("Product Yield vs Selectivity", fontsize=16)
plt.xlim(0, 100)
plt.ylim(0, 100)
# tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.axhline(0, color='k', lw=1)
plt.axvline(0, color='k', lw=1)

# save the figure
plt.tight_layout()
plt.savefig("figures/pareto_front.png", dpi=300, bbox_inches="tight")

plt.show()