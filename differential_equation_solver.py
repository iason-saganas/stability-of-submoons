import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G
from time import time
from typing import Union, List
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
mu_m_sm = get_standard_grav_parameter(hosting_body=moon, hosted_body=submoon, check_direct_orbits=False)
mu_p_m = get_standard_grav_parameter(hosting_body=planet, hosted_body=moon, check_direct_orbits=False)
mu_s_p = get_standard_grav_parameter(hosting_body=star, hosted_body=planet, check_direct_orbits=False)

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

# for i in np.linspace(moon.a0, moon.a,10):

y_init = [submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0]

# Finalize the system
planetary_system = bind_system_gravitationally(planetary_system=[star, planet, moon, submoon], use_initial_values=True)

list_of_all_events = [update_values, track_submoon_sm_axis_1, track_submoon_sm_axis_2,
                      track_moon_sm_axis_1, track_moon_sm_axis_2]

# evolve to 4.5 Bn. years
final_time = turn_billion_years_into_seconds(4.5)

# Solve the problem
sol_object = solve_ivp(fun=submoon_system_derivative, t_span=(0, final_time), y0=y_init, method="RK23",
                       args=(planetary_system, mu_m_sm, mu_p_m, mu_s_p), events=list_of_all_events)

time_points, _, _, _, _, _, _, status, message, success = unpack_solve_ivp_object(sol_object)

"""
# Unpack variables outputted by solution object.
(time_points, solution, t_events, y_events, num_of_eval, num_of_eval_jac, num_of_lu_decompositions, status, message,
 success) = unpack_solve_ivp_object(sol_object)

print("sol_object : ", sol_object)
custom_experimental_plot(time_points, solution, submoon_system_derivative)
"""

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
print("Success: ", success)

