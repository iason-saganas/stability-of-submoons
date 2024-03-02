import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G
from time import time
from utilities import *
from creation_of_celestial_bodies import *

print("Program start.\n")

'''
Please see the naming conventions in the file `getting_started_demo.py`.
----- ToDo: 
    -   Only simulate a priori stable systems!
    -   Check that masses are such, that eccentricities are very small.
'''

# Construct the base system
star, planet, moon, submoon = create_submoon_system(visualize_with_plot=False)

# Get standard gravitational parameters to input into the differential equations
mu_s_p = get_standard_grav_parameter(hosting_body=star, hosted_body=planet, check_direct_orbits=False)
mu_p_m = get_standard_grav_parameter(hosting_body=planet, hosted_body=moon, check_direct_orbits=False)
mu_m_sm = get_standard_grav_parameter(hosting_body=moon, hosted_body=submoon, check_direct_orbits=False)

# Construct the initial values of the system, which is not reflected in 'create_submoon_system'
planet_omega = 1/(5*3600)  # Giant Theia Impact => Days were 5 hours long
moon_distance = moon.a/2.05   # Also moon was apparently 17x closer when it formed
corresponding_orbit_frequency_moon = keplers_law_n_from_a_simple(moon_distance, mu_p_m)
moon_omega = corresponding_orbit_frequency_moon  # Moon has been tidally locked since birth!

submoon.a0 = submoon.a
moon.a0 = moon_distance
planet.a0 = planet.a

moon.omega0 = moon_omega
planet.omega0 = planet_omega
star.omega0 = star.omega

#for i in np.linspace(moon.a0, moon.a,10):

y_init = [submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0]

# Finalize the system
bind_system_gravitationally(planetary_system=[star, planet, moon, submoon], use_initial_values=True)


# Define system of differential equations
def derivative(t, y):
    """
    Calculates and returns the derivative vector dydt. The derivative vector contains the differential equations for

    a_m_sm
    a_p_m
    a_s_p
    omega_m
    omega_p
    omega_s

    in that order. See the equations at https://github.com/iason-saganas/stability-of-submoons/ .
    """
    # Tuple-Unpack the variables to track
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y

    # Convert the semi-major-axes into their corresponding orbit frequency. Necessary steps for signum function.
    n_m_sm = keplers_law_n_from_a_simple(a_m_sm, mu_m_sm)
    n_p_m = keplers_law_n_from_a_simple(a_p_m, mu_p_m)
    n_s_p = keplers_law_n_from_a_simple(a_s_p, mu_s_p)

    # Define the semi-major-axis derivatives
    a_m_sm_dot = get_a_factors(submoon) * np.sign(omega_m - n_m_sm) * a_m_sm ** (-11 / 2)
    a_p_m_dot = get_a_factors(moon) * np.sign(omega_p - n_p_m) * a_p_m ** (-11 / 2)
    a_s_p_dot = get_a_factors(planet) * np.sign(omega_s - n_s_p) * a_s_p ** (-11 / 2)

    # Define the spin-frequency derivatives
    omega_m_dot = np.round(-get_omega_factors(moon) * (np.sign(omega_m - n_p_m) * planet.mass ** 2 * a_p_m ** (-6)
                                              + np.sign(omega_m - n_m_sm) * submoon.mass ** 2 * a_m_sm ** (-6)), 10)
    omega_p_dot = (- get_omega_factors(planet) * (np.sign(omega_p - n_s_p) * star.mass ** 2 * a_s_p ** (-6)
                   + np.sign(omega_p - n_p_m) * moon.mass ** 2 * a_p_m ** (-6)))
    omega_s_dot = - get_omega_factors(star) * np.sign(omega_s - n_s_p) * planet.mass ** 2 * a_s_p**(-6)

    # Define and return the derivative vector
    dy_dt = [a_m_sm_dot, a_p_m_dot, a_s_p_dot, omega_m_dot, omega_p_dot, omega_s_dot]
    return dy_dt


"""
Note
---- 
Update semi-major-axes and spin-frequencies via class method. This is to be passed to the `events` argument
of `solve_ivp`.
Check if the Roche-Limit or Hill-Radius of the bodies is trespassed based on the updated semi-major-axis values.
If yes, terminate the program (terminal attribute set to True).
"""


def update_values(t, y):
    # track and update all relevant values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y

    star.update_spin_frequency_omega(omega_s)
    planet.update_spin_frequency_omega(omega_p)
    moon.update_spin_frequency_omega(omega_p)

    planet.update_semi_major_axis_a(a_s_p)
    moon.update_semi_major_axis_a(a_p_m)
    submoon.update_semi_major_axis_a(a_m_sm)

    # Use an infinity value, so to not actually activate the event
    return omega_s-np.inf


def track_submoon_sm_axis_1(t, y):
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    rl = submoon.get_current_roche_limit()
    return a_m_sm - rl


def track_submoon_sm_axis_2(t, y):
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    crit_a = submoon.get_current_critical_sm_axis()
    return a_m_sm - crit_a


def track_moon_sm_axis_1(t, y):
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    rl = moon.get_current_roche_limit()
    return a_p_m - rl


def track_moon_sm_axis_2(t, y):
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    crit_a = moon.get_current_critical_sm_axis()
    return a_p_m - crit_a


update_values.terminal = False
track_submoon_sm_axis_1.terminal = True
track_submoon_sm_axis_2.terminal = True
track_moon_sm_axis_1.terminal = True
track_moon_sm_axis_2.terminal = True

list_of_all_events = [update_values, track_submoon_sm_axis_1, track_submoon_sm_axis_2,
                      track_moon_sm_axis_1, track_moon_sm_axis_2]

# evolve to 4 Bn. years
final_time = turn_billion_years_into_seconds(4.5)

# Solve the problem
sol_object = solve_ivp(fun=derivative, t_span=(0, final_time), y0=y_init, method="RK23",
                       events=list_of_all_events)

# Unpack variables outputted by solution object.
(time_points, solution, t_events, y_events, num_of_eval, num_of_eval_jac, num_of_lu_decompositions, status, message,
 success) = unpack_solve_ivp_object(sol_object)

if status == 0:
    print("Stable input parameters.")
elif status == 1:
    print("A termination event occurred")
elif status == -1:
    print("Numerical error.")
else:
    raise ValueError("Unexpected outcome.")

print("Number of time steps taken: ", len(time_points))
print("Status: ", status)
print("Message: ", message)
# print("sol_object : ", sol_object)
# custom_experimental_plot(time_points, solution, derivative)

