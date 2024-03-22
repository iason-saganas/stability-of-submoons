import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G
from utilities import *
from time import time
from creation_of_celestial_bodies import *

print("Program start.\n")

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

analytical_lifetime = analytical_lifetime_one_tide(a_0=toy_satellite.a, a_c=1.01*toy_satellite.a,
                                                   hosted_body=toy_satellite)
analytical_lifetime_billions = turn_seconds_to_years(analytical_lifetime, "Billions")

print("ANALYTICAL ----- : ")
print("Time it takes for moon to evolve outwards by 1%: ",
      analytical_lifetime_billions, " billion years.\n")

print("NUMERICAL ----- : ")


# Define semi-major-axis derivative
def derivative(t, y):
    a_s, = y  # tuple unpack the variables to track
    derivative_a_s = get_a_derivative_factors_experimental(toy_satellite) * a_s**(-11/2)
    dydt = [derivative_a_s]
    return dydt


def custom_event(t, y):
    # track when y = 0.5% bigger than y0
    return y - 1.005*toy_satellite.a


# Don't terminate integration algorithm if custom callback event is triggered
custom_event.terminal = False

# Define initial values and employ `solve_ivp`
y_init = [toy_satellite.a]
for method, col in zip(['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'],
                       ["red", "green", "blue", "yellow", "black", "purple"]):
    timer_start = time()
    sol_object = solve_ivp(fun=derivative, t_span=(0, analytical_lifetime), y0=y_init, method=method,
                           events=custom_event)
    timer_end = time()
    # Unpack variables outputted by solution object.
    (time_points, solution, t_events, y_events, num_of_eval, num_of_eval_jac, num_of_lu_decompositions, status, message,
     success) = unpack_solve_ivp_object(sol_object)

    tracked_event_time = turn_seconds_to_years(t_events[0][0], "Billions")
    plt.vlines(tracked_event_time, ymin=1, ymax=1.01, color=col, label=f"Tracked event, method {method}")
    print(f"Time at which callback was triggered (y grew 0.5%): {tracked_event_time} Bn. years.")

    # Unpack semi-major-axis solution and get semi-major-axis value at last timestamp.
    semi_major_axis_solution = solution[0]
    last_registered_solution_value = semi_major_axis_solution[len(semi_major_axis_solution)-1]
    print(f"Numerical solution with method {method}: a({analytical_lifetime_billions} bn years) = "
          f"{np.round(100*last_registered_solution_value/toy_satellite.a-100, 4)}, a_0.\n"
          f"Integration was done in {len(time_points)} time steps "
          f"and took {np.round(timer_end-timer_start, 6)} seconds.\n\n")

    plt.plot(turn_seconds_to_years(time_points, "Billions"), semi_major_axis_solution / toy_satellite.a,
             "-", color=col)
    plt.plot(turn_seconds_to_years(time_points, "Billions"), semi_major_axis_solution / toy_satellite.a,
             ".", color=col, label=f"Solution points method {method}")

print("\nNote that the equation solved here was not the same as what we need to do in two ways:"
      "\n1.   It is not a system of six coupled differential equations. This may cause numerical problems."
      "\n2.   The term signum(relative_frequency) was assumed to be one. In reality, this may be a source of numerical"
      " stiffness. ")

plt.legend()
plt.title("Numerical Solution of one-tide equation")
plt.xlabel("Time in billions of years")
plt.ylabel("Relative satellite semi-major-axis")
plt.show()
