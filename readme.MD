
# Stability of Submoons

In this code, we numerically integrate the equations of motions governing the tidal dynamics in a 'four-nested-bodies system', that is to say a submoon system: A submoon orbiting a moon, orbiting a planet, orbiting a star. 


## Differential Equations for Orbit and Spin Frequencies

![App Screenshot](https://raw.githubusercontent.com/iason-saganas/stability-of-submoons/master/media/Screenshot%20of%20DFE.png)

These are six ordinary, first-order, non-linear differential equations (quantities exponentiated to a non-trivial power and signum function).
We will probably go about doing this using the 'solve_ivp' function from SciPy. Although we would have to think which
algorithm would work best in this situation. A tree visualizing the variable dependencies between these equations is shown.

![App Screenshot](https://raw.githubusercontent.com/iason-saganas/stability-of-submoons/master/media/Screenshot%20of%20DFE.png)