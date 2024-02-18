import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G

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

'''
