import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G
from utilities import *
from helper_classes import *

'''
Variables and functions naming conventions
------------------

Function as well as variable names are written in lowercase, seperated by underscores.
Class names are written in the CamelCase convention.
Instances of classes, like functions and variables, are also written in snake_case. 
File names are written in snake_case.

-   a:          Semi-major-axis
-   n:          Orbit-frequency
-   omega:      Spin-Frequency 
-   k:          Second tidal love number. Index "2" is omitted here.
-   Q:          Quality factor. Written uppercase as an exception.
-   mu:         Standard gravitational parameter between two bodies, i.e. mu = G(m_1+m_2)

These variables can be indexed via underscores with their descriptive indices: 

-   sm:         Submoon
-   m:          Moon
-   p:          Planet
-   s:          Star

Furthermore, these variables can be indexed by "0" to indicate an initial value.
E.g., the variable representing the initial value of the semi-major-axis of the submoon is written as

a_sm_0.

By initial value we mean: a_sm_0 = a_sm(t=0).

The gravitational parameter should always be specified as 

mu_sm_m,

for the submoon-moon system.

First, we construct the solution of a moon planet system and check whether the sign function in the differential 
equation is a source of stiffness, i.e. causes numerical problems. 

'''

toy_satellite, toy_planet = create_toy_satellite_and_planet()


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