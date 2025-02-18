import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# from style_components.matplotlib_style import *

# Interesting plot for the paper, showing how the error increases unboundedly when the real solution diverges and
# that the point of divergence comes a bit later because step-size is not immediately tremendously
# (enough) decreased => deviations
# Also plot in a third plot the relative difference between time steps

# This file also contains the ideas on how to detect divergences. I should test these ideas further and create
# multiple arrays for which the gradient etc is calculated to then finally show the λ ratio over the divergent / non-
# divergent plots

# Explanation: By checking whether or not there was significant time progress in the numeric solution (i.e. in the
# finally returned array `time_points` by essentially checking the relative time step length, I can see whether or not
# ANY of the six solutions has a divergence.
# How to detect which one actually do? First of all, calculate the point at which the relative step length becomes
# excessively small (e.g. 1e-3).
# This marks a transition point.
# Then, calculate the gradient of a solution.
# Note λ = gradient_directly_before_transition / gradient_after_transition = grad[transition_index-1] / gradient[-1].
# If this ratio is sufficiently small (e.g. 1e-6), the numeric solution at hand has indeed diverged at time_points[-1].

# Constants
k1 = -4.050037492091401e+31  # given k1 constant
c_1 = -2.6325243698594107e+32  # additional constant for comparison function
a_m_sm_0 = 24042816.662809804  # initial semimajor-axis
t_final = 1.41912e+17  # simulation time in seconds


# Define the differential equation
def ode_function(t, a):
    return k1 * a**(-11/2)


# Analytical solution function
def analytical_solution(t):
    return (13 / 2 * (k1 * t + (2 / 13) * (a_m_sm_0 ** (13 / 2))))**(2 / 13)


# Comparison function
def ana_comparison(t):
    return (c_1 * t + a_m_sm_0**(13/2))**(2/13)


# Solve the ODE using solve_ivp with the Radau method
sol = solve_ivp(ode_function, [0, t_final], [a_m_sm_0], method='Radau', rtol=1e-1)

# Extract the numerical solution and corresponding times
numerical_solution = sol.y[0]
numerical_time_points = sol.t

# Calculate the analytical solution over a linspace from 0 to t_final * 1.5
# extended_time_points = np.concatenate((np.linspace(np.max(numerical_time_points) * 0.8, np.max(numerical_time_points) * 1.5, 1000), numerical_time_points))
extended_time_points = numerical_time_points
analytical_values_extended = analytical_solution(extended_time_points)

# Calculate the comparison values
comparison_values_extended = ana_comparison(extended_time_points)

# Calculate the relative difference at numerical time points
relative_difference = np.abs(numerical_solution - analytical_solution(numerical_time_points)) / analytical_solution(numerical_time_points) * 100

# Convert time to billions of years (1 billion years = 3.154e+16 seconds)
time_in_billion_years_numerical = numerical_time_points / 3.154e+16
time_in_billion_years_extended = extended_time_points / 3.154e+16

# Convert numerical and analytical solutions to SLU
slu_conversion = 19220000.0  # meters
numerical_solution_slu = numerical_solution / slu_conversion
analytical_values_extended_slu = analytical_values_extended / slu_conversion
comparison_values_extended_slu = comparison_values_extended / slu_conversion

# Plotting
plt.subplot(2, 1, 1)
plt.plot(time_in_billion_years_numerical, numerical_solution_slu, 'o-', color='red', label='Numerical Solution (Radau)', lw=2)  # Red line and dots for numerical solution
plt.plot(time_in_billion_years_extended, analytical_values_extended_slu, 'D', color='orange', label='Analytical Solution', lw=0)  # Orange diamonds for analytical solution
plt.xlabel('Time (Billions of Years)')
plt.ylabel('Semimajor Axis (SLU)')
plt.title('Numerical vs. Analytical vs. Comparison Solution')
plt.legend()
plt.grid()

# Set y-axis limits
plt.ylim(0.13, 1.6)

plt.subplot(2, 1, 2)
plt.plot(time_in_billion_years_numerical, relative_difference)
plt.xlabel("time")
plt.ylabel("relative deviation from analytical solution")

# Print numerical and analytical solutions with relative differences
explicit = False
print("relative_differences ", relative_difference, " mean: ", np.nanmean(relative_difference))
if explicit:

    for t, num_sol, rel_diff in zip(numerical_time_points, numerical_solution, relative_difference):
        ana_sol = analytical_solution(t)
        print(f'Time: {t} s, Numerical: {num_sol} m, Analytical: {ana_sol} m, Relative Difference: {rel_diff}%')


def neighbouring_diffs(arr):
    return np.diff(arr) / arr[:-1]  # cut ot first element


# Gradients
gradient_ana = np.gradient(analytical_values_extended_slu, time_in_billion_years_extended)
gradient_num = np.gradient(numerical_solution_slu, time_in_billion_years_extended)
gradient_num_raw = np.gradient(numerical_solution, numerical_time_points)
relative_time_step_differences = neighbouring_diffs(numerical_time_points)  # cut ot first element
print("gradient_ana ", gradient_ana)
print("gradient_num ", gradient_num)
print("relative time-step differences t_i+1-t_i ", relative_time_step_differences)
print("time points themselves: ", numerical_time_points)

print("\n")
print("gradient neighbouring_diffs ", neighbouring_diffs(gradient_num))

plt.show()

transition_indices = np.where(relative_time_step_differences < 1e-3)[0]  # too small time-steps indicating some barrier
transition_idx = transition_indices[0]
transition_time_point = numerical_time_points[transition_idx]

# grad_after_trans = np.mean(gradient_num_raw[transition_idx+1:transition_idx+1+10])  # Mean Gradient after transition (10 samples)
# grad_before_trans = np.mean(gradient_num_raw[transition_idx-1-10:transition_idx-1])  # Mean Gradient before transition (10 samples

# Or alternative, possibly better formulation:

grad_after_trans = gradient_num_raw[-1]
grad_before_trans = gradient_num_raw[transition_idx-1]

print("Gradient after transition (last sample) :", grad_after_trans)
print("Gradient before transition (first sample) :", grad_before_trans)

print("Their ratio, λ (before / after): ", grad_before_trans/grad_after_trans)

plt.vlines(transition_time_point, -3000, 1, label="Detected transition time")

plt.plot(numerical_time_points, gradient_num_raw, "b.")
plt.xlabel("t in bne years")
plt.ylabel("gradient in m/s")
plt.legend()
plt.show()


def neighbouring_diffs(arr):
    return np.diff(arr) / arr[:-1]  # cut ot first element


def detect_divergence(t_arr, y_arr):
    """
    Detects whether y_arr does shows indeed signs of suspected divergence by:
    Assuming that the points in the t_arr get closer and closer together, and finding a transition point after which the
    relative change from one x point to the next is miniscule (e.g. 1e-3 threshhold) and only gets smaller.

    This cuts the t_arr into two regions: A possibly broad region that is suspected to have a relatively small gradient
    and a narrow region with suspected very high region.

    λ is defined as the ratio between the gradient in the first region (represented by the gradient sample directly
    before the transition region) and the gradient in the second region (represented by the last gradient sample
    possible, which is taken to be a good characteristic since the second region is suspected to be narrow).

    If λ < 1e-6, return true (diverged), else false (non-diverged)

    :param t_arr:
    :param y_arr:
    :return:
    """
    relative_progress = neighbouring_diffs(numerical_time_points)  # in the t-direction
    try:
        tr_idx = np.where(relative_progress < 1e-3)[0][0]
    except IndexError:
        # No transition point was detected.
        return False
    grad = np.gradient(y_arr, t_arr)
    lambda_ratio = grad[tr_idx-1] / grad[-1]
    if lambda_ratio < 1e-6:
        return True
    else:
        return False







