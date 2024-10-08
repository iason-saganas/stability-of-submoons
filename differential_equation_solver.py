from utilities import *
from creation_of_celestial_bodies import *

# Please see the naming conventions in the file `getting_started_demo.py`.
# ToDo:
#    -   Only simulate a priori stable systems! Currently, `bind_system_gravitationally` checks that the distance ratio
#        between the bodies is .3, such that the bodies don't crash into each other as a proxy for stability.
#        If not, an error is raised.
#    -   Check that masses are such, that eccentricities are very small. Currently, `bind_system_gravitationally` checks
#        that the mass ratio between hosted and hosting body is .1.
#        If not, an error is raised.

# Construct the base system
star, planet, moon, submoon = create_submoon_system(visualize_with_plot=False)

# Get standard gravitational parameters to input into the differential equations
mu_m_sm = get_standard_grav_parameter(hosting_body=moon, hosted_body=submoon, check_direct_orbits=False)
mu_p_m = get_standard_grav_parameter(hosting_body=planet, hosted_body=moon, check_direct_orbits=False)
mu_s_p = get_standard_grav_parameter(hosting_body=star, hosted_body=planet, check_direct_orbits=False)

# Construct the initial values of the system, which is not reflected in 'create_submoon_system'
star_omega = star.omega
planet_omega = 1 / (5 * 3600) * 4  # Giant Theia Impact => Days were 5 hours long
moon_omega = keplers_law_n_from_a_simple(moon.a, mu_p_m) / 2  # Tidally locking moon

# Set the omega initial values
moon.update_spin_frequency_omega(moon_omega)
planet.update_spin_frequency_omega(planet_omega)
star.update_spin_frequency_omega(star_omega)

# The values of the semi-major-axes of the planet, moon and submoon are updated dynamically in each iteration.
# Order:
# [submoon.a, moon.a, planet.a, moon.omega, planet.omega, star.omega],
PLACEHOLDER = None
y_init = [PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, moon.omega, planet.omega, star.omega]

n_pix_planet = 10
n_pix_moon = 10
n_pix_submoon = 10

# Set the resolution, i.e., how many simulations to do and other parameters of the simulation
result = solve_ivp_iterator(n_pix_planet=n_pix_planet, n_pix_moon=n_pix_moon, n_pix_submoon=n_pix_submoon,
                            y_init=y_init, planetary_system=[star, planet, moon, submoon], debug_plot=True,
                            list_of_std_mus=[mu_m_sm, mu_p_m, mu_s_p], upper_lim_planet=30, lower_lim_planet=0.5)

# showcase_results(result)
