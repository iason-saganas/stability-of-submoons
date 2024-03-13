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
moon_omega = corresponding_orbit_frequency_moon  # The moon has been tidally locked since birth!

# Set the found initial values of the system via attribute dot notation
submoon.a0 = submoon.a
moon.a0 = moon_distance
planet.a0 = planet.a

moon.omega0 = moon_omega
planet.omega0 = planet_omega
star.omega0 = star.omega

# Hard copy of all s.m.axes and spin frequencies of the system. Their values are updated dynamically in each iteration.
# Through the copy, `reset_to_default` can be called, which reassigns the values given on creation by
# `create_submoon_system`.
copy_of_default_vars = [submoon.a, moon.a, planet.a, moon.omega, planet.omega, star.omega]
names = ['submoon.a', 'moon.a', 'planet.a', 'moon.omega', 'planet.omega', 'star.omega']

# The index of the variable to vary. E.g., 1 for `moon.a`.
index_to_vary = 1
quantity_to_vary = copy_of_default_vars[index_to_vary]

# Define the resolution, i.e., how many simulations to do
n_pix = 100

result = solve_ivp_iterator(key=names[index_to_vary], index_to_vary=index_to_vary, upper_lim=quantity_to_vary,
                            n_pix=100, y_init=[submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0],
                            y_default=copy_of_default_vars, planetary_system=[star, planet, moon, submoon],
                            list_of_std_mus=[mu_m_sm, mu_p_m, mu_s_p])

print("Result: ", result)

