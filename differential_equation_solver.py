from collections import defaultdict
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
moon_distance = moon.a   # Also moon was apparently 17x closer when it formed
corresponding_orbit_frequency_moon = keplers_law_n_from_a_simple(moon_distance, mu_p_m)
moon_omega = corresponding_orbit_frequency_moon  # The moon has been tidally locked since birth!

# Set the found initial values of the system via update class method
submoon.update_semi_major_axis_a(submoon.a)
moon.update_semi_major_axis_a(moon_distance)
planet.update_semi_major_axis_a(planet.a)

moon.update_spin_frequency_omega(moon_omega)
planet.update_spin_frequency_omega(planet_omega)
star.update_spin_frequency_omega(star.omega)

# Hard copy of all s.m.axes and spin frequencies of the system. Their values are updated dynamically in each iteration.
# Through the copy, `reset_to_default` can be called, which reassigns the values given on creation by
# `create_submoon_system`.
y_init = [submoon.a, moon.a, planet.a, moon.omega, planet.omega, star.omega]

# Define the resolution, i.e., how many simulations to do and other parameters of the simulation
n_pix_moon = 20
n_pix_submoon = 5
low_lim = moon.get_current_roche_limit()
up_lim = moon.get_current_critical_sm_axis()
result = solve_ivp_iterator(n_pix_moon=n_pix_moon, n_pix_submoon=n_pix_submoon, y_init=y_init,
                            planetary_system=[star, planet, moon, submoon],
                            list_of_std_mus=[mu_m_sm, mu_p_m, mu_s_p])

hits = result[0]
hit_counts = defaultdict(int)
for hit in hits:
    hit_counts[hit] += 1

print("\n\n--------RESULTS--------\n\n")
print("\n------ Stability states (-1: Termination, NUM ER: Numerical error, +1: Longevity):")
for key, value in hit_counts.items():
    print(f'{key}: {value}')
print("\n------")

print("\n------ Histogram of termination reasons:")
for key, value in result[1].items():
    print(f'{key}: {value}')
print("\n------")

print("\n------ Longest lifetime objects (Second element is lifetime in years):")
max_subarray = max(result[2], key=lambda x: x[1])
index_of_longest_lifetime_array = result[2].index(max_subarray)
print(result[2][index_of_longest_lifetime_array])
