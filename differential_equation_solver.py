from utilities import *
from creation_of_celestial_bodies import *
import numpy as np

# Please see the naming conventions in the file `getting_started_demo.py`.

CASE_A = True  # The star spins slower than the planet, which spins slower than the moon
CASE_B = False  # The star spins slower than the moon, which spins slower than the planet.
CASE_C = False  # The planet spins slower than the star, which spins slower than the moon.
CASE_D = False  # The planet spins slower than the moon, which spins slower than the star.
CASE_E = False  # The moon spins slower than the star, which spins slower than the planet.
CASE_F = False  # The moon spins slower than the planet, which spins slower than the star.

all_cases = np.array([CASE_A, CASE_B, CASE_C, CASE_D, CASE_E, CASE_F])

name_case = [["Case_A", " (star spins slower than the planet, which spins slower than the moon)"],
             ["Case_B", " (star spins slower than the moon, which spins slower than the planet)"],
             ["Case_C", " (planet spins slower than the star, which spins slower than the moon)"],
             ["Case_D", " (planet spins slower than the moon, which spins slower than the star)"],
             ["Case_E", " (moon spins slower than the star, which spins slower than the planet)"],
             ["Case_F", " (moon spins slower than the planet, which spins slower than the star)"]]

which_true = np.where(all_cases == True)[0]
if len(which_true) > 1:
    raise ValueError("Choose one.")

case = name_case[which_true[0]][0]
case_descr = name_case[which_true[0]][1]

if case == "Case_A":
    star_rot_period_days = 20
    planet_rot_period_hours = 50
    moon_rot_period_hours = 24
elif case == "Case_B":
    star_rot_period_days = 20
    planet_rot_period_hours = 24
    moon_rot_period_hours = 10 * 24  # 10 days
elif case == "Case_C":
    star_rot_period_days = 20
    planet_rot_period_hours = 50 * 24  # 50 days
    moon_rot_period_hours = 10 * 24  # 10 days
elif case == "Case_D":
    star_rot_period_days = 20
    planet_rot_period_hours = 50 * 24  # 50 days
    moon_rot_period_hours = 40 * 24  # 40 days
elif case == "Case_E":
    star_rot_period_days = 20
    planet_rot_period_hours = 10
    moon_rot_period_hours = 50 * 24  # 50 days
elif case == "Case_F":
    star_rot_period_days = 5
    planet_rot_period_hours = 10*24  # 10 days
    moon_rot_period_hours = 30*24  # 30 days
else:
    raise ValueError


arr = []
earth_mass = 5.972e24  # in kg

# sm_masses = np.linspace(2, 10, 10)*earth_mass

# Logarithmic regime
log_part = np.logspace(-5, -1, 15)  # More points in this range

# Linear regime
linear_part = np.linspace(10**-2, 1, 5)  # Starts from ~10^-2

# Merge the arrays
sm_masses = np.concatenate((log_part, linear_part))*earth_mass  # test multiple masses

# test the default mass instead
# sm_masses = [None]

for sm_mass in sm_masses:

    print("Checking submoon mass ", sm_mass, " if `None` -> Default")

    # Construct the base system, don't forget to change the `case_prefix` variable!
    star, planet, moon, submoon = create_warm_jupiter_submoon_system(P_rot_star_DAYS=star_rot_period_days,
                                                              P_rot_planet_HOURS=planet_rot_period_hours,
                                                              P_rot_moon_HOURS=moon_rot_period_hours,
                                                                     custom_sm_mass=sm_mass)
    # case_prefix = "earth_like"
    case_prefix = "warm_jupiter_like"

    # Get standard gravitational
    # parameters to input into the differential equations
    mu_m_sm = get_standard_grav_parameter(hosting_body=moon, hosted_body=submoon, check_direct_orbits=False)
    mu_p_m = get_standard_grav_parameter(hosting_body=planet, hosted_body=moon, check_direct_orbits=False)
    mu_s_p = get_standard_grav_parameter(hosting_body=star, hosted_body=planet, check_direct_orbits=False)


    # The values of the semi-major-axes of the planet, moon and submoon are updated dynamically in each iteration.
    # Order
    # [submoon.a, moon.a, planet.a, moon.omega, planet.omega, star.omega],
    PLACEHOLDER = None
    y_init = [PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, moon.omega, planet.omega, star.omega]

    n_pix_planet = 7
    n_pix_moon = 7
    n_pix_submoon = 7


    # test_cases = [(2, 2, 2),
	# (2, 2, 3),
	# (2, 2, 4),
    # ]

    # for tc in test_cases:

    # Set the resolution, i.e., how many simulations to do and other parameters of the simulation
    result = solve_ivp_iterator(n_pix_planet=n_pix_planet, n_pix_moon=n_pix_moon, n_pix_submoon=n_pix_submoon,
                                y_init=y_init, planetary_system=[star, planet, moon, submoon], debug_plot=False,
                                list_of_std_mus=[mu_m_sm, mu_p_m, mu_s_p], upper_lim_planet=3, lower_lim_planet=None,
                                case_prefix=case_prefix, further_notes=case+case_descr,
                                analyze_iter=False, specific_counter=None, force_tanh_approx=False, unequal_support=False)

    arr.append(find_fraction_of_stable_iterations(result))


arr = np.array(arr)
print("Mass array in Earth masses: ", sm_masses/earth_mass)
print("Resulting stable iterations: ", arr)
np.savetxt("data_storage/arrays/stable_iteration_fraction_submoon_mass_variation.txt", arr)

# showcase_results(result, suppress_text=False, plot_initial_states=True, plot_final_states=True,
#                   save=False, filename=case, earth_like=False)
