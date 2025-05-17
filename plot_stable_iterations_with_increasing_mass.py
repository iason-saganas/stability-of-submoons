import matplotlib.pyplot as plt
from style_components.matplotlib_style import *
import numpy as np

arr = np.loadtxt("data_storage/arrays/stable_iteration_fraction_submoon_mass_variation.txt")

# Logarithmic regime
log_part = np.logspace(-5, -1, 15)  # More points in this range

# Linear regime
linear_part = np.linspace(10**-2, 1, 5)  # Starts from ~10^-2

# Merge the arrays
sm_masses = np.concatenate((log_part, linear_part))

indcs_to_remove = [10, 11,12] # for visual purposes

arr = np.delete(arr, indcs_to_remove)
sm_masses = np.delete(sm_masses, indcs_to_remove)

plt.plot(sm_masses, arr, color="black", marker="x", lw=0
         , markersize=10)

plt.xlabel(r"Submoon mass $[M_{\oplus}]$")
plt.ylabel(r"Fraction of stable iterations $[\%]$")

plt.tight_layout()
# plt.savefig("data_storage/figures/stable_iteration_fraction_as_a_function_of_sm_mass.png")
plt.show()