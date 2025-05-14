import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(["science", "no-latex", "grid"])

font_size = 18

times = np.linspace(0, 30, 100)

def flow_rate(t):
    if t < 5:        
        return 5
    
    else:
        return 0.2
    
flow_rates = [flow_rate(t) for t in times]

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))

# make the background of the plot red for the first 5 minutes
ax[0].axvspan(0, 5, color='red', alpha=0.4)
ax[0].axvspan(25, 30, color='red', alpha=0.4)
# blue otherwise
ax[0].axvspan(5, 25, color='blue', alpha=0.4)

# in the first axis, plot the flow rates
ax[0].plot(times, flow_rates, label='Flow rate 1', color='blue', linewidth=5)
ax[0].set_xlabel('Experiment Time / min', fontsize=font_size)
ax[0].set_ylabel('Flow Rate / mL/min', fontsize=font_size)
# ax[0].legend(fontsize = font_size)

# set x-axis limits
ax[0].set_xlim(0, 30)

# set xtick only at 25
ax[0].set_xticks([25])
ax[0].set_yticks([0.2, 5])

# in the second axis, plot the corresponding residence times, simply a ramp between 1 and 21
t_rts = np.linspace(5, 25, 100)
rts = [(t - 5) / 20 * 20 + 1 for t in t_rts]

# add dead times
equivs = [1] + rts + [21]
t_equivs = [0] + list(t_rts) + [30]

ax[1].axvspan(0, 5, color='red', alpha=0.4)
ax[1].axvspan(25, 30, color='red', alpha=0.4)
ax[1].axvspan(5, 25, color='blue', alpha=0.4)

ax[1].plot(t_equivs, equivs, color='blue', linewidth=5)
ax[1].set_xlabel('Experiment Time / min', fontsize=font_size)
ax[1].set_ylabel('Residence Time / min', fontsize=font_size)

ax[1].set_xlim(0, 30)

# set xtick only at 25
ax[1].set_xticks([25])
ax[1].set_yticks([1, 21])

# in the third axis, plot the product yield vs residence time
equivs = np.linspace(1, 21, 50)
equiv_calc = (equivs - 1) / 20 + 1
# use a log
product_yield = [(np.log((equiv - 0.98) * 20) + 1)/5 for equiv in equiv_calc]
# add noise
product_yield = [yield_ + np.random.normal(0, 0.0125 / 2) for yield_ in product_yield]
# multiply by 100 to get percentage
product_yield = [yield_ * 100 for yield_ in product_yield]

ax[2].scatter(equivs, product_yield)
ax[2].set_xlabel('Residence Time / min', fontsize=font_size)
ax[2].set_ylabel('Product Yield / %', fontsize=font_size)

# set x-ticks at 1 and 2
ax[2].set_xticks([0, 10, 20])
# tick font size
ax[0].tick_params(axis='both', which='major', labelsize=font_size)
ax[1].tick_params(axis='both', which='major', labelsize=font_size)
ax[2].tick_params(axis='both', which='major', labelsize=font_size)

# save the figure
plt.tight_layout()
plt.savefig("figures/rt_illustration.pdf", dpi=300, bbox_inches="tight")

plt.show()