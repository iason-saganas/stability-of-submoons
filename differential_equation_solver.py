import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G
from utilities import *
from creation_of_celestial_bodies import *

print("Program start.\n")

'''
Please see the naming conventions in the file `getting_started_demo.py`.
----- ToDo: 
    -   Only simulate a priori stable systems!
    -   Check that masses are such, that eccentricities are very small.
'''

# Construct system
star, planet, moon, submoon = create_toy_submoon_system(visualize_with_plot=True)

# Get standard gravitational parameters
mu_s_p = get_standard_grav_parameter(hosting_body=star, hosted_body=planet, check_direct_orbits=False)
mu_p_m = get_standard_grav_parameter(hosting_body=planet, hosted_body=moon, check_direct_orbits=False)
mu_m_sm = get_standard_grav_parameter(hosting_body=moon, hosted_body=submoon, check_direct_orbits=False)

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
    omega_m_dot = 0
    omega_p_dot = 0
    omega_s_dot = 0

    # Define and return the derivative vector
    dydt = [a_m_sm_dot]
    return dydt


"""
So, I have to construct it like this: 

y = [omega_1, omega_2, omega_3, ... a_1, a_s ... and so on] and then: 

def derivative(t,y # time must come first, solution vector second): 
    dydt = a list containing all derivatives of the variables in y (ordered of course) and the elements in y can be
    accessed by indexing, 
    i.e. omega_1, omega_2, omega_3, ... a_1, a_s ... and so on = y 
    deriv 1 = some function of omega_1 and a_1
    deriv 2 = some function of omega_2 and omega_3 or someting 
    and then 
    dydt = [deriv1, deriv2, ... etc.]
    return dydt ! 

y_init = [initial value list for ] y

    Good ressource: https://pundit.pratt.duke.edu/wiki/Python:Ordinary_Differential_Equations/Examples

"""