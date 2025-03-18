import numpy as np
from scipy.constants import G
from warnings import warn as raise_warning
import pandas as pd
from typing import Union, List
from scipy.integrate import solve_ivp
from style_components.matplotlib_style import *
import matplotlib.pyplot as plt
import pickle
from style_components.voxel_plotter import plot_3d_voxels_initial_states, plot_3d_voxels_final_states
from matplotlib.ticker import FuncFormatter
import datetime
import time

# Delete and import instead from style_components
"""plt.style.use("dark_background")
red = (0.74, 0.1, 0.1, 1)
light_red = (0.74, 0.1, 0.1, 0.4)
blue = (0, 0.37, 0.99, 1)
light_blue = (0.42, 0.8, 0.93, 0.9)
lighter_blue = (0.42, 0.8, 0.93, 0.3)
lightest_blue = (0.42, 0.8, 0.93, 0.1)
green = (0.23, 0.85, 0.25, 1)
light_green = (0.23, 0.85, 0.25, 0.4)
yellow = (0.89, 0.89, 0)"""

AU = 1.496e11
LU = 384.4e6  # 'Lunar unit', the approximate distance between earth and earth's moon
SLU = LU / 20  # 'Sublunar unit', a twentieth of a lunar unit

# TEST
# t_final = 3597776267807147
# extended_time_points = np.linspace(0, t_final * 1.5, 1000)
# analytical_values_extended = calc(extended_time_points)

__all__ = ['check_if_direct_orbits', 'keplers_law_n_from_a', 'keplers_law_a_from_n', 'keplers_law_n_from_a_simple',
           'get_standard_grav_parameter', 'get_hill_radius_relevant_to_body', 'get_critical_semi_major_axis',
           'get_roche_limit', 'analytical_lifetime_one_tide', 'dont', 'get_solar_system_bodies_data',
           'CelestialBody', 'turn_seconds_to_years', 'get_a_derivative_factors_experimental', 'get_a_factors',
           'get_omega_derivative_factors_experimental', 'get_omega_factors', 'unpack_solve_ivp_object',
           'turn_billion_years_into_seconds', 'bind_system_gravitationally', 'state_vector_plot',
           'submoon_system_derivative', 'update_values', 'track_sm_m_axis_1', 'track_sm_m_axis_2',
           'track_m_p_axis_1', 'track_m_p_axis_2', 'reset_to_default', 'solve_ivp_iterator', 'showcase_results',
           'pickle_me_this', 'unpickle_me_this']


# noinspection StructuralWrap
class Tracker:
    """
    A simple class in whose attributes the state vector can be stored in each iteration by calling inside of
    the `update_values` function.
    This serves as the datastructure for the stiffness detection scheme.
    `tracker = Tracker()` and `tracker.y` will get an array whose elements are arrays representing the state vector
    at that time point with the entries `a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y` or their derivatives.

    Attributes:

   self.t (np.array):           An array whose elements are floats, representing the time points.
   self.y (np.array):           An array whose elements represent the state vector at the time point associated with the index
                                of the element.
   self.dy_dt (np.array):       As `self.y` but containing the derivative of the state vector at any given time.
   self.eta_chain (np.array)    An array of floats that represent the evolution of the `stiffness_coefficent` over time.
   self.signs_body (np.array)   Array of floats that represent the sign of sgn(Ω_i - n_j) that is a factor in the semi-
                                major-axis derivative of the body.
                                Each step in the solution provides one value to these arrays.

   NOTE : The array tracker.t must NOT be exactly equal to the final returned time points by the solver.

    """

    def __init__(self):
        self.t = None
        self.y = None
        self.dy_dt = None
        self.eta_chain = []
        self.signs_sm = []  # Ω_m - n_sm_m
        self.signs_m = []  # Ω_p - n_p_m
        self.signs_p = []  # Ω_s - n_s_p
        self.iter_counter = None  # Sth. like (1, 2, 3)

    def add(self, new_y_point, new_dy_dt_ypoint, new_t_point):
        # Standard add function for time, state and derivative of state
        y = np.array([new_y_point])
        dydt = np.array([new_dy_dt_ypoint])
        t = np.array([new_t_point])
        if self.y is None:
            self.y = y
            self.dy_dt = dydt
            self.t = t
        else:
            self.y = np.vstack((self.y, y))
            self.dy_dt = np.vstack((self.dy_dt, dydt))
            self.t = np.append(self.t, t)

    def add_eta(self, eta):
        # add function for the stiffness coefficients
        self.eta_chain.append(eta)

    def add_signs(self, triple):
        """
        The signs to attach to the chains of `self.signs_sm` etc. for each time point.
        This method should be called each time the derivative is calculated to calculate the next step (i.e. inside
        of an event function), not if the purpose was to approximate the jacobian (inside the derivative itself), other-
        wise the arrays will be longer than the number of time points in the solution.

        triple[0] has to be np.sign(Ω_m - n_sm_m)
        triple[1] has to be np.sign(Ω_p - n_p_m)
        triple[2] has to be np.sign(Ω_s - n_s_p)

        :param triple: A tuple containing the sign changes.
        :return:
        """
        self.signs_sm.append(triple[0])
        self.signs_m.append(triple[1])
        self.signs_p.append(triple[2])

    def clear(self):
        # Clear function
        self.y = None
        self.dy_dt = None
        self.t = None
        self.eta_chain = []
        self.signs_sm = []
        self.signs_m = []
        self.signs_p = []

    def get_sign_changes(self):
        """
        Calculates the number of sign changes, from one element to the next, of the data stored in the arrays
        `self.signs_sm, self.signs_m, self.signs_p` and returns those numbers.

        :return: list,  A list that contains the number of sign changes for the submoon, moon and planet.
        """
        res = [calculate_sign_changes(arr) for arr in [self.signs_sm, self.signs_m, self.signs_p]]
        return res


# Initialize global tracker object
tracker = Tracker()


def check_if_direct_orbits(hosting_body: 'CelestialBody', hosted_body: 'CelestialBody'):
    """
    Checks whether the hosted (e.g. submoon) and hosting body (e.g. moon) are in direct orbits of each other
    by comparing their hierarchy numbers. Throws value errors if not.

    :parameter hosting_body:    CelestialBody,      The body that is being orbited by `hosted_body`.
    :parameter hosted_body:     CelestialBody,      The body that orbits `hosting_body`.

    """
    if hosted_body.hn == 1:
        raise ValueError('A celestial number of hierarchy 1 (star) cannot be a hosted body.')
    distance = np.abs(hosted_body.hn - hosting_body.hn)
    if distance != 1:
        raise ValueError(f'The inputted bodies cannot be in direct orbit of each other since their hierarchy numbers'
                         f' are too far apart. Expected difference between hierarchy numbers: 1. Got: {distance}')


# Define a function to format the y-axis labels with desired decimal places
def format_func(value, tick_number):
    return f'{np.round(value, 2)}'


def keplers_law_n_from_a(hosting_body: 'CelestialBody', hosted_body: 'CelestialBody') -> float:
    """
    Gets `hosted_body`'s current semi-major-axis and converts it to the corresponding orbit frequency using Kepler's
    Third Law.

    :parameter hosting_body:    CelestialBody,      The body that is being orbited by `hosted_body`.
    :parameter hosted_body:     CelestialBody,      The body that orbits `hosting_body`.

    """
    check_if_direct_orbits(hosting_body, hosted_body)
    mu = get_standard_grav_parameter(hosting_body=hosting_body, hosted_body=hosted_body, check_direct_orbits=False)
    n = mu ** (1 / 2) * hosted_body.a ** (-3 / 2)
    return n


def keplers_law_n_from_a_simple(a: float, mu: float) -> float:
    """
    Same functionality as `keplers_law_n_from_a`, but doesn't fetch the CelestialBody objects since mu needs to be
    provided directly.

    :parameter a:    float,      The semi-major-axis to convert to the corresponding orbit frequency.
    :parameter mu:   float,      The standard gravitational parameter to use for the conversion.

    """
    n = mu ** (1 / 2) * a ** (-3 / 2)
    return n


def keplers_law_a_from_n(hosting_body: 'CelestialBody', hosted_body: 'CelestialBody') -> float:
    """
    Gets `hosted_body`'s current orbit-frequency and converts it to the corresponding semi-major-axis using Kepler's
    Third Law.

    :parameter hosting_body:    CelestialBody,      The body that is being orbited by `hosted_body`.
    :parameter hosted_body:     CelestialBody,      The body that orbits `hosting_body`.

    """
    check_if_direct_orbits(hosting_body, hosted_body)
    mu = get_standard_grav_parameter(hosting_body=hosting_body, hosted_body=hosted_body, check_direct_orbits=False)
    a = mu ** (1 / 3) * hosted_body.n ** (-2 / 3)
    return a


def get_standard_grav_parameter(hosting_body: 'CelestialBody', hosted_body: 'CelestialBody',
                                check_direct_orbits=True) -> float:
    """
    Gets the standard gravitational parameter mu = G(m_1+m_2).

    :parameter hosting_body:            CelestialBody,      The body that is being orbited by `hosted_body`.
    :parameter hosted_body:             CelestialBody,      The body that orbits `hosting_body`.
    :parameter check_direct_orbits:     bool,               (Optional) Whether to perform the sanity check specified
                                                            by the function `check_if_direct_orbits`. For example
                                                            not needed inside the class method
                                                            `CelestialBody.get_standard_grav_parameters()`
                                                            since sanity check there already performed during
                                                            initialization. Default value: True.

    :return: mu:            float,              The calculated mu value.
    """
    if check_direct_orbits:
        check_if_direct_orbits(hosting_body=hosting_body, hosted_body=hosted_body)
    return G * (hosting_body.mass + hosted_body.mass)


def simple_barycenter(mass_ratio_small_to_big, radius_big, distance_small):
    """
    This function takes

    :param mass_ratio_small_to_big:     The mass ratio of a satellite to its hosting body,
    :param radius_big:                  The radius of the hosting body and
    :param distance_small:              The semi-major axis of the satellite,

    to return the

    :return:        approximate position of the barycenter in hosting body's radii units away from the hosting body's
                    geometric center. The returned variable, called `alpha` is calculated as in `check_barycentre_moon`.
    """
    return mass_ratio_small_to_big * 1/radius_big  * distance_small


def get_hill_radius_relevant_to_body(hosted_body: 'CelestialBody') -> float:
    """
    Gets the hill radius that is relevant to `hosted_body`, i.e. the gravitational sphere of influence exerted by the
    body that `hosted_body` orbits.

    Let i, j, k represent hierarchy numbers, with i < j < k, i.e. `i` is the most 'un-nested' body and `k` is very
    nested: `k` orbits `j` orbits `i`.

    Then, the hill-radius is defined as

    r_h_k = a_j * (m_j / (3*m_i))**(1/3)

    :parameter hosted_body: CelestialBody,                  The body to find the relevant hill radius for by getting
                                                            information on its hosting body and ITS hosting body again.
    :return: r_h:  float,                                   The found hill-radius
    """
    if hosted_body.hn <= 2:
        raise_warning("WARNING: Can the hill-radius between the planet and star be defined?")
        raise ValueError(f"You can't find the hill radius of the body with hierarchy number {hosted_body.hn},"
                         f" since information about the hosting's body hosting's body needs to be gotten, i.e."
                         f" a body of the hierarchy number {hosted_body.hn - 2}. ")

    k = hosted_body
    j = k.hosting_body
    i = j.hosting_body

    r_h = j.a * (j.mass / (3 * i.mass)) ** (1 / 3)
    return r_h


def get_critical_semi_major_axis(hosted_body: 'CelestialBody') -> float:
    """
    Gets the critical semi-major-axis of `hosted_body` after which it escapes the gravitational influence of its
    primary. This is a fraction of the hill-radius, i.e.

    a_crit = f*r_h.

    According to this paper

    https://iopscience.iop.org/article/10.3847/1538-3881/ab89a7/pdf,

    f = 0.4 r_h_p   for a moon and
    f = 0.33 r_h_m  for a submoon.

    :param hosted_body: CelestialBody,            The body to find the critical semi-major-axis for.
    :return: a_crit: float,                       The found critical semi-major-axis.
    """
    if hosted_body.hn == 3:
        f = 0.4
    elif hosted_body.hn == 4:
        f = 0.33
    else:
        raise_warning("WARNING: Can the hill-radius between the planet and star be defined?")
        raise ValueError(f"You can't find the hill radius of the body with hierarchy number {hosted_body.hn},"
                         f" since information about the hosting's body hosting's body needs to be gotten, i.e."
                         f" a body of the hierarchy number {hosted_body.hn - 2}. ")

    a_crit = f * get_hill_radius_relevant_to_body(hosted_body)
    return a_crit


def get_roche_limit(hosted_body: 'CelestialBody') -> float:
    """
    Gets the distance to the primary at which `hosted_body` is disintegrated by tidal forces.
    Let j be the `hosted_body` and i its hosting body. Then, the formula is

    a_l = R_j * (3*m_i / m_j )**(1/3)

    :parameter hosted_body: CelestialBody,     The body to find the roche-limit for.
    :return: a_l: float,                       The roche limit at which `hosted body` is disintegrated.
    """
    j = hosted_body
    i = j.hosting_body
    a_l = j.R * (3 * i.mass / j.mass) ** (1 / 3)
    return a_l


def analytical_lifetime_one_tide(a_0: float, a_c: float, hosted_body: 'CelestialBody') -> float:
    """
    Calculates the time it takes for the semi-major-axis to reach `a_i` ,starting from `a_0` using the inputted set of
    parameters describing the system. This represents the analytical formula for the lifetime T in a one-tide-system,
    given by Murray and Dermott equation (4.213).

    Let j represent the satellite and i its hosting body. Then,

    T = 2/13 * a_0^(13/2) * ( 1-(a_i/a_0) ^ (13/2) ) * ( 3k_{2i} / Q_i * R_i^5 * m_j * (G/m_i)^(1/2) )^(-1)

    :parameter a_0:         float,              The initial semi-major-axis of the satellite j.
    :parameter a_c:         float,              The semi-major-axis value of j to evolve to.
    :parameter hosted_body: CelestialBody,      The satellite j to evolve.
    :return: T:             float,              The analytically calculated time it took for the evolution.
    """
    j = hosted_body
    i = j.hosting_body
    left_hand_side = 2 / 13 * a_c ** (13 / 2) * (1 - (a_0 / a_c) ** (13 / 2))
    right_hand_side = 3 * i.k / i.Q * (G / i.mass) ** (1 / 2) * i.R ** 5 * j.mass
    T = left_hand_side / right_hand_side
    return T


def dont():
    """
    Do nothing function.
    """
    pass


def get_solar_system_bodies_data(file_to_read: str, name_of_celestial_body: str = '', physical_property: str = '',
                                 print_return: bool = False) -> Union['SpecialDict', float]:
    """
    Reads the file contents 'constants/planets_solar_system.txt'.

    If only `planet_name` is provided, data for that planet will be returned and can then be accessed via dot notation
    of the following shorthands:

    columns_shorthand = ['name', 'm', 'a', 'd', 'T_orbit_days', 'e', 'rho', 'T_rotation_hours', 'k', 'Q'],

    standing for the objects name, mass, semi-major-axis, diameter, orbital period in days, eccentricity, density,
    rotation period in hours, second tidal love number and quality factor respectively, all in SI units if not just
    specified otherwise.

    Then:

    sun = get_solar_system_bodies_data("sun")
    semi_major_axis_sun = sun.a


    If `planet_name` and `physical_property` is provided, a float will be returned that represent the queried
    physical property.

    :param file_to_read: str,                  The file to read from, for example 'planets_solar_system.txt'.
    :param name_of_celestial_body: str,        One of: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus or Neptune.
    :param physical_property: str,  One of: Mass, Semi-major-axis, Diameter, Orbital-Period, Orbital-eccentricity,
                                            Density, Rotation-Period, 2nd-Tidal-Love-Number, Quality-factor.
    :param print_return: bool,      Print what is returned.
    """
    df = pd.read_csv(file_to_read)
    whole_row_mode = (name_of_celestial_body != '' and physical_property == '')
    specific_value_mode = (name_of_celestial_body != '' and physical_property != '')

    def turn_to_float_if_possible(x):
        if isinstance(x, Union[float, int, np.int64]):
            return x
        elif isinstance(x, str):
            represents_digit = x.replace(".", "", 1).isdigit()
            if represents_digit:
                return float(x)
            else:
                return x
        else:
            raise_warning("Unexpected situation arisen inside function `get_solar_system_bodies_data`.")

    if whole_row_mode:
        row = df[df['Body'] == name_of_celestial_body]
        row_nice_representation = row.iloc[0]

        columns = row_nice_representation.index.values  # Array
        # prints: ['Body' 'Mass-(kg)' 'Semi-major-axis-(m)' 'Diameter-(m)' 'Orbital-Period-(days)'
        # 'Orbital-eccentricity' 'Density-(kg/m^3)' 'Rotation-Period-(hours)' '2nd-Tidal-Love-Number-(Estimate)'
        # 'Quality-factor-(Estimate)'] => Now we hardcode shorthands.
        columns_shorthand = ['name', 'm', 'a', 'd', 'T_orbit_days', 'e', 'rho', 'T_rotation_hours', 'k', 'Q']
        values = [turn_to_float_if_possible(x) for x in row_nice_representation.values]  # Array

        # Create canonical python dict and then instantiate custom class in which dict values can be accessed via dot
        # notation (I prefer it that way)
        normal_dict = dict(zip(columns_shorthand, values))
        special_dict = SpecialDict(normal_dict)

        print(row_nice_representation) if print_return else dont()
        return special_dict

    elif specific_value_mode:
        row = df[df['Body'] == name_of_celestial_body]
        row_nice_representation = row.iloc[0]
        all_column_names = df.columns.tolist()
        user_short_hand = physical_property
        selected_index = 0
        for index, column_name in enumerate(all_column_names):
            if user_short_hand in column_name:
                selected_index = index
                break
        res = turn_to_float_if_possible(row_nice_representation[all_column_names[selected_index]])
        print(res) if print_return else dont()
        return res
    else:
        print(df)


class CelestialBody:

    def __init__(self, mass: float, density: float, semi_major_axis: Union[float, None], spin_frequency: float,
                 love_number: float, quality_factor: float, descriptive_index: str, name: str, hierarchy_number: int,
                 hosting_body: Union['CelestialBody', None]):

        """
        Base class representing a celestial body with its various properties set as class attributes.

        Because a parameter 'hosting_body' is needed, when defining all celestial bodies, the star is the very
        first body that needs to be instantiated such that it can be passed as a parameter to the planet, which in turn
        can be passed to the moon's definition etc.

        On instantiation, the parameters `semi_major_axis` and `spin_frequency` should represent the initial values.

        Attributes:
        ----------------

        :parameter mass:                Float,          Body's mass.
        :parameter density:             Float,          The body's mean density.
        :parameter semi_major_axis:     Union[float, None],   Value for the semi-major-axis. On instantiation, this should be the
                                                        semi-major-axis initial value `a_0` and may then be updated through
                                                        the method 'update_semi_major_axis'.
        :parameter spin_frequency:      Float,          Value for the spin-frequency. On instantiation, this should be the
                                                        spin-frequency initial value `omega_0` and may then be updated
                                                        through the method 'update_spin_frequency'.
        :parameter love_number:         Float,          The second tidal love number associated with the body's rheology.
        :parameter quality_factor:      Float,          The quality factor associated with the body's rheology.
        :parameter descriptive_index:   Str,            A descriptive index shorthand for the body, e.g. "sm" for submoon.
        :parameter name:                Str,            The name for the body, e.g. "submoon".
        :parameter hierarchy_number:    Int,            The hierarchy number corresponding to the body's position in the
                                                        nested body system. 1 for star, 2 for the planet, 3 for the moon,
                                                        4 for submoon.
        :parameter hosting_body:        Union[float, None],  The body that `self` orbits, i.e. the hosting body.

        Methods:
        ----------------
        update_semi_major_axis_a:       Updates the semi-major-axis based on a value.
        update_spin_frequency_omega:    Updates the spin-frequency based on a value.
                                        Semi-major-axis. After initialization, `self.n` can be used instead.
                                        `Self` and its hosts as specified by `self.hosting_body`


        Properties
        (Can be accessed via dot notation of a class instance like attributes but are defined via a distinct class method
        instead of inside `__init__()`) :
        ----------------
        n:                              The current orbit-frequency calculated from the current semi-major-axis using
                                        Kepler's Third Law.
        mu:                             The standard gravitational parameter between `self` and the body that `self` orbits,
                                        specified by `self.hosting_body`.
        R:                              Assuming a spherical volume, calculates the mean radius based on mass and density.
        I:                              Assuming I = 2/5 M R^2, calculates the inertial moment.

        """

        self.a0 = None  # Can be updated dynamically when setting up the solve_ivp.
        self.omega0 = None  # Can be updated dynamically when setting up the solve_ivp.
        self.mass = mass
        self.rho = density
        self.omega = spin_frequency
        self.k = love_number
        self.Q = quality_factor
        self.descriptive_index = descriptive_index
        self.name = name
        self.hn = hierarchy_number
        self.oscillation_counter = 0  # How many times the signum function flipped the sign the semi-major-axis
        # or spin frequency derivative

        if hierarchy_number == 1:
            # Star has no hosting body.
            self.hosting_body = None
            self.a = None
        else:
            self.a = semi_major_axis
            try:
                check_if_direct_orbits(hosting_body=hosting_body, hosted_body=self)
                self.hosting_body = hosting_body
            except ValueError as err:
                raise ValueError(f"The hosting body's hierarchy number does not match with the hierarchy number of "
                                 f" the instantiated celestial body '{self.name}': Error message: ", err)

    def __str__(self):
        head_line = f"\nCelestialBody `{self.name}` \n"
        seperator = "---------------"
        properties = "\n"
        for name, val in self.__dict__.items():
            if name == 'hosting_body' and val is not None:
                # to not print all of this again for the hosting body
                properties += f"hosting_body: CelestialBody `{val.name}`\n"
            else:
                properties += f"{name}: {val}\n"
        finisher = "\n---------------\n"
        return head_line + seperator + properties + finisher

    def update_semi_major_axis_a(self, update_value):
        """
        Updates the semi-major-axis of the celestial body.

        :parameter update_value:    float,      The semi-major-axis value to update to.

        """
        self.a = update_value

    def update_spin_frequency_omega(self, update_value):
        """
        Updates the spin-frequency of the celestial body.

        :parameter update_value:    float,      The spin-frequency value to update to.

        """
        self.omega = update_value

    @property
    def n(self) -> float:
        """
        Uses Kepler's Third Law to get and return the current orbit frequency of the body `n` from the current
        semi-major-axis `a`. For this, the body that hosts `self` is needed.
        Since the `@property` decorator is used, this can be accessed like an attribute, `self.n`

        :return orbit_frequency:    float,              The calculated orbit frequency.

        """
        n = keplers_law_n_from_a(hosting_body=self.hosting_body, hosted_body=self)
        return n

    @property
    def mu(self) -> float:
        """
        Gets the standard gravitational parameter mu = G(m_1+m_2) where m_1 is self.m and m_2 is mass of the
        hosting body.
        No need to check whether hosting and hosted bodies are really in direct orbits of each other since this
        was already checked in the initialization.
        Since the `@property` decorator is used, this can be accessed like an attribute, `self.mu`

        :return: mu:            float,              The calculated mu value.
        """
        return get_standard_grav_parameter(hosting_body=self.hosting_body, hosted_body=self, check_direct_orbits=False)

    @property
    def R(self) -> float:
        """
        Gets the mean circular radius of `self` based on its mass.
        mass = rho * V  => V = mass / rho
        r =  ( 3*V / (4*np.pi) ) **(1/3) = ( 3 * mass / rho / (4*np.pi) ) **(1/3)

        :return: r:            float,              The calculated mean radius value.
        """
        r = (3 * self.mass / self.rho / (4 * np.pi)) ** (1 / 3)
        return r

    @property
    def I(self) -> float:
        """
        Gets the inertial moment based on the assumption of a rotation sphere of radius R with mass M:

        I = 2/5 M R^2
        ToDo: Introduce alpha parameter to increase accuracy
        :return: float,     The calculated inertial moment
        """
        I = 2 / 5 * self.mass * self.R ** 2
        return I

    def get_current_roche_limit(self) -> float:
        # Distance to its hosting body, at which `self` is disintegrated
        rl = get_roche_limit(self)
        return rl

    def get_current_critical_sm_axis(self) -> float:
        # Distance to its hosting body, at which `self` escapes its gravitational influence
        crit_sm_axis = get_critical_semi_major_axis(self)
        return crit_sm_axis


def turn_seconds_to_years(seconds: float, keyword: str = "Vanilla") -> float:
    """
    Converts seconds into years ("Vanilla"), millions ("Millions") or billions ("Billions") of years.
    :param seconds: float,  The seconds to convert.
    :param keyword: str,    Either "Vanilla", "Millions" or "Billions".
    :return: float, the converted time. Result is rounded to two decimal places.
    """
    conversion_factor = 3600 * 24 * 365
    seconds_in_years = seconds / conversion_factor
    if keyword == "Vanilla":
        return np.round(seconds_in_years, 2)
    elif keyword == "Millions":
        return np.round(seconds_in_years / 10 ** 6, 2)
    elif keyword == "Billions":
        return np.round(seconds_in_years / 10 ** 9, 2)
    else:
        raise_warning("Something unexpected occured inside function `turn_seconds_to_years`.")


def get_a_derivative_factors_experimental(hosted_body: 'CelestialBody') -> float:
    """

    :parameter hosted_body:             CelestialBody,      The satellite.
    :return: float, the calculated multiplicative factor for the sm-axis derivative.
    """
    j = hosted_body
    i = j.hosting_body
    res = 3 * i.R ** 5 * j.mu ** (1 / 2) * i.k * j.mass / (i.Q * i.mass)
    return res


def get_omega_derivative_factors_experimental(body: 'CelestialBody') -> float:
    """
    Represents the common, multiplicative factors of the omega dot equations, see

    https://github.com/iason-saganas/stability-of-submoons

    for the equations
    :param body:        The body that indexes the quantities to catch in an omega dot equation. See again the equations.
    :return:    float, the calculated multiplicative factors
    """
    i = body
    res = 3 * G * i.R ** 5 * i.k / (2 * i.Q * i.I)
    return res


# Aliases for `get_a_derivative_factors_experimental` and `get_omega_derivative_factors_experimental`
get_a_factors = get_a_derivative_factors_experimental
get_omega_factors = get_omega_derivative_factors_experimental


def unpack_solve_ivp_object(solve_ivp_sol_object):
    """
    Unpacks some variables deemed to be relevant from the returned object by the function `solve_ivp`
    """
    time_points = np.array(solve_ivp_sol_object.t)
    solution = np.array(solve_ivp_sol_object.y)
    t_events = solve_ivp_sol_object.t_events
    y_events = solve_ivp_sol_object.y_events
    num_of_eval = np.array(solve_ivp_sol_object.nfev)
    num_of_eval_jac = np.array(solve_ivp_sol_object.njev)
    num_of_lu_decompositions = np.array(solve_ivp_sol_object.nlu)
    status = np.array(solve_ivp_sol_object.status)
    message = np.array(solve_ivp_sol_object.message)
    success = np.array(solve_ivp_sol_object.success)
    return (time_points, solution, t_events, y_events, num_of_eval, num_of_eval_jac,
            num_of_lu_decompositions, status, message, success)


def turn_billion_years_into_seconds(num: float) -> float:
    """
    :param num: float,      Then number of billions of years to turn into seconds
    """
    return num * 1e9 * 365 * 24 * 3600


class SpecialDict:
    """
    This is a helper class that uses a normal python dictionary as input and stores its key values pairs as attributes,
    such that values can be accessed via dot-notation of their keys, which I prefer.

    test_dict = {'blah': 42}
    special_dict = SpecialDict(test_dict)
    val = special_dict.blah
    # val == 42

    """

    def __init__(self, d):
        self._dict = d

    def __getattr__(self, attr):
        if attr in self._dict:
            return self._dict[attr]
        else:
            raise AttributeError(f"'SpecialDict' object has no attribute '{attr}'")


def bind_system_gravitationally(planetary_system: List['CelestialBody'], use_initial_values: bool = False,
                                verbose: bool = True):
    """
    This 'finalizes' an instantiated planetary system: It performs the following sanity checks:

    1.  Are the mass ratios of the inputted bodies small? If not raise ValueError: Assumption of circularity is broken.
    2.  Are the distances between the bodies such that one body does not crash into the other?
    (3.  Not implemented: Does the system have basic gravitational stability? => Lagrangian Filter?)

    If all the checks run successfully, the system is returned, otherwise an exception is raised.

    :param planetary_system: List['CelestialBody'],        The list containing instances of 'CelestialBody' that make up
                                                           the planetary system. Does not have to be ordered.
    :param use_initial_values: bool,                       Optional. If false, the stored a and omega values inside the
                                                           `CelestialBody` instances represent the initial values of the
                                                           problem. If that is not that case, the attributes `a0` and
                                                           `omega0` have to be attached by hand (moon.a0 = 0.03 e.g.)
                                                           and the `initial_values` parameter set to True to indicate
                                                           that the attributes `.a0` and `.omega0` should be used
                                                           instead of `.a` and `.omega`.
    :param verbose: bool,                                  Whether to print the calculated distance and mass ratios.

    If the semi-major-axes and spin-frequencies that are stored in the `CelestialBody` instances already represent the
    initial values, the parameter `initial_values` does not have to be provided.

    :return planetary_system: list,                         The inputted planetary system, if all goes well.

    """
    if not use_initial_values:
        raise_warning("\nFunction `bind_system_gravitationally()` --> Performing mass and distance ratio sanity "
                      "checks for the inputted system.\n Check whether the stored values in the inputted "
                      "`CelestialBody` instances \n are actually the initial values for the system. \n"
                      "If not, set the parameter `initial_values` to `True` \n and attach the attributes `a0` and"
                      "`omega0` to each `CelestialBody` instance by hand.\n")

    hn_list = [body.hn for body in planetary_system]

    # Perform mass and distance ratios check
    # m_ratio = .1
    m_ratio = .25  # For testing purposes
    a_ratio = .3
    for index, hn in enumerate(hn_list):
        if hn == 1:
            # No hosting body to check mass or semi-major-axis against if it's on the first hierarchy level
            pass
        else:
            hosted_body = planetary_system[index]
            hosting_body = hosted_body.hosting_body

            # No longer needed, since appropriateness of mass ratio is now checked dynamically inside the
            # `solve_ivp_iterator` by tracking whether the distance between the larger body's geometric center
            # and a subsystem's barycenter lies outside the larger body's radius.

            # mass_ratio = hosted_body.mass / hosting_body.mass
            # if mass_ratio > m_ratio:
            #    raise ValueError(f"\n Exception inside 'bind_system_gravitationally' function: \n"
            #                     f"Mass ratio between bodies {hosting_body.name} and {hosted_body.name} is too big.\n "
            #                     f"Detected mass ratio: {mass_ratio}. Threshold: {m_ratio} "
            #                     f"(assumption that needs to be checked!).\n Absolute mass values: {hosting_body.mass} "
            #                     f"and {hosted_body.mass} respectively. \n")
            # elif verbose:
            #    print(f"Mass ratio sanity check between bodies {hosting_body.name} and {hosted_body.name} passed.\n "
            #          f"            Detected mass ratio: {mass_ratio}. Threshold: {m_ratio}.\n\n")



            if hosted_body.hn == 2:
                # Semi-major-axis of hosting.body is not defined => no semi-major-axis ratio
                pass
            else:
                if not use_initial_values:
                    distance_ratio = hosted_body.a / hosting_body.a
                else:
                    distance_ratio = hosted_body.a0 / hosting_body.a0
                if distance_ratio > a_ratio:
                    raise ValueError(f"\n Exception inside 'bind_system_gravitationally' function: \n"
                                     f"Distance ratio between bodies {hosting_body.name} and {hosted_body.name} is too "
                                     f"big. \nDetected distance ratio: {distance_ratio}. Threshold: {a_ratio} "
                                     f"(assumption that needs to be checked!).\n Absolute distance values: "
                                     f"{hosting_body.a} and {hosted_body.a} respectively. ")
                elif verbose:
                    print(f"Distance ratio sanity check between bodies {hosting_body.name} and {hosted_body.name} "
                          f"passed. \n"
                          f"            Detected semi-major-axis ratio: {distance_ratio}. Threshold: {a_ratio}.\n\n")

    return planetary_system


def state_vector_plot(time_points, solution, derivative, planetary_system, list_of_mus, show=True, save=False,
                      plot_derivatives=True, old_state=None):
    """
    :param plot_derivatives: Whether or not to plot the derivatives of each function as well
    :param save: Whether to save a figure of this plot
    :param show: Whether or not to show this plot
    :param list_of_mus: The list of standard gravitational parameters
    :param planetary_system: The list of CelestialBodies.
    :param time_points: The time points at which solutions were found
    :param solution: numpy array, the y solution coming out of 'solve_ivp'
    :param derivative: The derivative (callable)
    :param old_state: An additional run to add to each subplot.
    Was implemented to compare approximate numerical solutions (tanh.
    approximation) to exact numerical solutions (np.sign).
    Needs to contain:

    time_points_old, solution_old, derivative_old = old_state

    :return:
    """

    time_norm = 3600 * 24 * 365 * 1e9  # turn seconds into Gyr's
    time_normed = time_points / time_norm

    # fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12,8))
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
    # solution[:6] contains: a_sm, a_m, a_p, omega_m, omega_p, omega_s, but `axes` is ordered as
    # a_sm, omega_m, a_m, omega_p, a_p, omega_s; so we reorder the `axes` object.
    dy_dt = np.array(derivative(time_points, solution, planetary_system, *list_of_mus))
    evolutions = solution[:6]  # a_sm = solution[:6][0]
    derivatives = dy_dt[:6]
    ordering = np.array([0, 2, 4, 1, 3, 5])
    reordered_axes = axes.flatten()[ordering]

    y_labels = [r"$a_{\mathrm{sm}}(t)$", r"$a_{\mathrm{m}}(t)$", r"$a_{\mathrm{p}}(t)$",
                r"$\Omega_{\mathrm{m}}(t)$", r"$\Omega_{\mathrm{p}}(t)$", r"$\Omega_{\mathrm{s}}(t)$"]
    legend_labels = ["SLU", r"LU or $\mathrm{\mu}$Hz ", r"AU or $\mathrm{\mu}$Hz", r"$\mathrm{\mu}$Hz",
                     r"$\mathrm{\mu}$Hz", r"$\mathrm{\mu}$Hz"]
    der_n = turn_billion_years_into_seconds(1e-3)  # derivative norm, have to be multiplied to get radians or meters per
    # Myrs as unit
    y_normalizations = [[SLU, 1 / der_n], [LU, 1 / der_n], [AU, 1 / der_n], [1 / 1e6, 1 / der_n], [1 / 1e6, 1 / der_n],
                        [1 / 1e6, 1 / der_n]]
    # `y_normalizations` elements are [evol norm, deriv norm].
    # 1/factor gets the factor multiplied instead of divided.
    # frequencies will be in microhertz, semi-major-axis in SLU, LU or AU and derivatives in meters or radians per Myr.
    colors = [red, green, blue, green, blue, yellow]
    show_legend_labels = [True, True, True, False, False, True]

    # extended_time = np.sort(np.concatenate((np.linspace(max(time_points)*0.9, max(time_points)*1.2, 1000), time_points)))
    extended_time = time_points
    extended_time_normed = extended_time / time_norm
    # evol_anas = semi_major_axes_analytical_solution(extended_time, solution.T[0], planetary_system, list_of_mus)
    # evol_anas.extend([None] * 3)  # No analytic solution for spin frequencies calculated
    evol_anas = np.array([None] * 6)  # If you don't wish to plot analytic evolutions uncomment this line
    handles, labels = [], []

    if old_state is None:
        evolutions_old, derivatives_old = [[None]*6]*2
        time_normed_old = None
    else:
        time_points_old, solution_old, derivative_old = old_state
        dy_dt_old = np.array(derivative_old(time_points_old, solution_old, planetary_system, *list_of_mus))
        evolutions_old = solution_old[:6]
        # derivatives_old = dy_dt_old[:6]
        derivatives_old = [None]*6  # don't want to plot these derivatives
        time_normed_old = time_points_old / time_norm

    lst = [reordered_axes, evolutions, evolutions_old, colors, y_labels, y_normalizations, legend_labels,
           show_legend_labels, derivatives, derivatives_old, evol_anas]

    zip_object = zip(*lst)

    for el in zip_object:

        (ax, evolution, evolution_old, color, y_label, norms, legend_label, show_legend_label,
         derivative, derivative_old, evol_ana) = el

        y_norm = norms[0]
        dy_dt_norm = norms[1]
        evol = evolution / y_norm
        der = derivative / dy_dt_norm

        ax.plot(time_normed, evol, "x", color=color, lw=0, label=legend_label, markersize=5)  # Plot solution with x
        plot_objects = ax.plot(time_normed, evol, color=color, lw=2, ls="-", label=legend_label, markersize=0)  # with -
        # ax2 = ax.twinx()
        # ax2.plot(time_normed, np.gradient(evol, time_points), "x", color="black", lw=0, markersize=4, )
        # Plot gradient
        if evolution_old is not None:
            evol_old = evolution_old / y_norm
            ax.plot(time_normed_old, evol_old, "x", color="purple", lw=0, markersize=5)  # Plot solution with x
            ax.plot(time_normed_old, evol_old, "--", color="purple", lw=2, markersize=0)  # Plot solution with --

        # Plot analytic
        if evol_ana is not None:
            evol_ana = evol_ana / y_norm
            ax.plot(extended_time_normed, evol_ana, "D", color="orange", markersize=2)

        # Plot derivatives
        if plot_derivatives:
            ax2 = ax.twinx()
            ax2.plot(time_normed, der, color=color, lw=0, markersize=2, marker=".")
            if derivative_old is not None:
                der_old = derivative_old / dy_dt_norm
                ax2.plot(time_normed, der_old, color="purple", lw=2, ls="--", markersize=0)

        ax.set_ylabel(y_label)
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        # ax2.yaxis.set_major_formatter(FuncFormatter(format_func))
        ax.set_ylim(min(evol) - 0.1, max(evol) + 0.1)  # in the units chosen (AU, LU, SLU and μHz), the y-scale of the
        # solution is in the same order of mag or larger than 0.1. Setting this ylim is sensible since microbehaviour
        # of the solution under this scale is not of interest.
        if show_legend_label:
            handles.extend(plot_objects)
            labels.append(plot_objects[0].get_label())

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Increase this value to add more space at the bottom
    fig.text(0.5, 0.05, 'Time passed in Billions of years', ha='center', va='center', fontsize=24, fontweight="bold")
    fig.align_ylabels(axes[:, 0])  # First column
    fig.align_ylabels(axes[:, 1])  # Second column

    # Adjust bottom margin for space for the legend
    plt.subplots_adjust(top=0.85)  # Increase the top margin to make space for the legend
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), ncol=3, fontsize=12)

    # Increase vertical space between columns
    plt.subplots_adjust(wspace=0.5)

    # Finally, insert information about the used units
    info = (r"$\mathrm{AU}$ astronomical unit, $\mathrm{LU}$ 'lunar unit' (mean earth-moon-distance), " + "\n" +
            r"$\mathrm{SLU}$ 'sub-lunar unit' ($1/20$th of a $\mathrm{LU}$). Number of steps taken: " +
            str(len(time_normed)) + ".")
    fig.text(0.45, 0.974, info, ha='left', va='top', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.5'))

    if save:
        now = datetime.datetime.now()
        plt.savefig("data_storage/figures/" + str(now) + ".png", dpi=30)
        plt.close(fig)
    if show:
        plt.show()
        plt.close(fig)


def c_sign(expr, t=False, k=1e7):
    # Custom sign function, if t=True, np.sign gets replaced by hyperbolic tangent approximation of `expr`
    if t:
        return np.tanh(k*expr)
    else:
        return np.sign(expr)


# Define system of differential equations
def submoon_system_derivative(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    """
    Calculates and returns the derivative vector dy_dt. The derivative vector contains the differential equations for

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

    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system

    # Convert the semi-major-axes into their corresponding orbit frequency. Necessary steps for signum function.
    n_m_sm = keplers_law_n_from_a_simple(a_m_sm, mu_m_sm)
    n_p_m = keplers_law_n_from_a_simple(a_p_m, mu_p_m)
    n_s_p = keplers_law_n_from_a_simple(a_s_p, mu_s_p)

    # Define the semi-major-axis derivatives
    a_m_sm_dot = get_a_factors(submoon) * c_sign(omega_m - n_m_sm) * a_m_sm ** (-11 / 2)
    a_p_m_dot = get_a_factors(moon) * c_sign(omega_p - n_p_m) * a_p_m ** (-11 / 2)
    a_s_p_dot = get_a_factors(planet) * c_sign(omega_s - n_s_p) * a_s_p ** (-11 / 2)

    # Define the spin-frequency derivatives
    omega_m_dot = (-get_omega_factors(moon) * (c_sign(omega_m - n_p_m) * planet.mass ** 2 * a_p_m ** (-6)
                                               + c_sign(omega_m - n_m_sm) * submoon.mass ** 2 * a_m_sm ** (
                                                   -6)))
    omega_p_dot = (- get_omega_factors(planet) * (c_sign(omega_p - n_s_p) * star.mass ** 2 * a_s_p ** (-6)
                                                  + c_sign(omega_p - n_p_m) * moon.mass ** 2 * a_p_m ** (-6)))
    omega_s_dot = - get_omega_factors(star) * c_sign(omega_s - n_s_p) * planet.mass ** 2 * a_s_p ** (-6)

    # Define and return the derivative vector
    dy_dt = [a_m_sm_dot, a_p_m_dot, a_s_p_dot, omega_m_dot, omega_p_dot, omega_s_dot]
    return dy_dt


# Define system of differential equations
def submoon_system_derivative_approximant(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    """
    Calculates and returns the derivative vector dy_dt. The derivative vector contains the differential equations for

    a_m_sm
    a_p_m
    a_s_p
    omega_m
    omega_p
    omega_s

    in that order. See the equations at https://github.com/iason-saganas/stability-of-submoons/ .
    This function approximates the sign function via np.tanh(k*x), where k is a big number.
    This approximation has much better numeric properties since it is smooth.
    """
    # Tuple-Unpack the variables to track
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y

    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system

    # Convert the semi-major-axes into their corresponding orbit frequency. Necessary steps for signum function.
    n_m_sm = keplers_law_n_from_a_simple(a_m_sm, mu_m_sm)
    n_p_m = keplers_law_n_from_a_simple(a_p_m, mu_p_m)
    n_s_p = keplers_law_n_from_a_simple(a_s_p, mu_s_p)

    # Define the semi-major-axis derivatives
    a_m_sm_dot = get_a_factors(submoon) * c_sign(omega_m - n_m_sm, t=True) * a_m_sm ** (-11 / 2)
    a_p_m_dot = get_a_factors(moon) * c_sign(omega_p - n_p_m, t=True) * a_p_m ** (-11 / 2)
    a_s_p_dot = get_a_factors(planet) * c_sign(omega_s - n_s_p, t=True) * a_s_p ** (-11 / 2)

    # Define the spin-frequency derivatives
    omega_m_dot = (-get_omega_factors(moon) * (c_sign(omega_m - n_p_m, t=True) * planet.mass ** 2 * a_p_m ** (-6)
                                               + c_sign(omega_m - n_m_sm, t=True) * submoon.mass ** 2 * a_m_sm ** (
                                                   -6)))
    omega_p_dot = (- get_omega_factors(planet) * (c_sign(omega_p - n_s_p, t=True) * star.mass ** 2 * a_s_p ** (-6)
                                                  + c_sign(omega_p - n_p_m, t=True) * moon.mass ** 2 * a_p_m ** (-6)))
    omega_s_dot = - get_omega_factors(star) * c_sign(omega_s - n_s_p, t=True) * planet.mass ** 2 * a_s_p ** (-6)

    # Before defining and returning derivative vector, check quality of tanh approximation
    # delta_angles = np.array([omega_m - n_p_m, omega_m - n_m_sm, omega_p - n_s_p, omega_p - n_p_m, omega_s - n_s_p])
    # relative_residuals = np.abs(np.sign(delta_angles)-c_sign(delta_angles, t=True))/np.sign(delta_angles)
    # delta_angle_names = np.array(["omega_m - n_p_m", "omega_m - n_m_sm", "omega_p - n_s_p", "omega_p - n_p_m", "omega_s - n_s_p"])
    # if np.any(relative_residuals > .1):
    #    where = np.where(relative_residuals > .1)
    #    angle_names = delta_angle_names[where]
    #    print("\t\tCAUTION: TANH APPROXIMATION DEVIATES FROM ACTUAL SIGN FOR: ", angle_names,
    #          " BY % ", relative_residuals[where], "; Real sign array ", np.sign(delta_angles), "\n\t used sign array:"
    #          , c_sign(delta_angles, t=True))

    # Define and return the derivative vector
    dy_dt = [a_m_sm_dot, a_p_m_dot, a_s_p_dot, omega_m_dot, omega_p_dot, omega_s_dot]
    return dy_dt


def semi_major_axes_analytical_solution(t: float, y0: np.array, planetary_system: List['CelestialBody'], list_of_mus):
    """
    Calculates an analytic solution of the semi-major-axes evolutions of submoon, moon and planet, assuming
    that no sign changes have occurred in their derivatives.
    Used to check simulations in selected cases where this is true.
    The calculations have been done by hand in a notepad.

    :param list_of_mus: list,   List of standard gravitational parameters that can be unpacked as below.
    :param t: float,            The time to calculate the solution for
    :param y0 np.array:         The initial state vector.
    :param planetary_system:    The planetary system under investigation
    :return: res: list:         A list containing the values of a_sm, a_m, a_p at the time t, in that order.
    """
    star, planet, moon, submoon = planetary_system
    mu_m_sm, mu_p_m, mu_s_p = list_of_mus
    a_m_sm_0, a_p_m_0, a_s_p_0, omega_m_0, omega_p_0, omega_s_0 = y0

    n_m_sm_0 = keplers_law_n_from_a_simple(a_m_sm_0, mu_m_sm)
    n_p_m_0 = keplers_law_n_from_a_simple(a_p_m_0, mu_p_m)
    n_s_p_0 = keplers_law_n_from_a_simple(a_s_p_0, mu_s_p)

    # Define constants
    c_1_prime = 3 * moon.R ** 5 * np.sign(omega_m_0 - n_m_sm_0) * np.sqrt(mu_m_sm) * moon.k * submoon.mass / (
                moon.Q * moon.mass)
    c_2_prime = 3 * planet.R ** 5 * np.sign(omega_p_0 - n_p_m_0) * np.sqrt(mu_p_m) * planet.k * moon.mass / (
                planet.Q * planet.mass)
    c_3_prime = 3 * star.R ** 5 * np.sign(omega_s_0 - n_s_p_0) * np.sqrt(mu_s_p) * star.k * planet.mass / (
                star.Q * star.mass)

    c_1 = 13 / 2 * c_1_prime
    c_2 = 13 / 2 * c_2_prime
    c_3 = 13 / 2 * c_3_prime

    a_m_sm_evolved = (c_1 * t + a_m_sm_0 ** (13 / 2)) ** (2 / 13)
    a_p_m_evolved = (c_2 * t + a_p_m_0 ** (13 / 2)) ** (2 / 13)
    a_s_p_evolved = (c_3 * t + a_s_p_0 ** (13 / 2)) ** (2 / 13)

    return [a_m_sm_evolved, a_p_m_evolved, a_s_p_evolved]


def reset_to_default(y: list, planetary_system: List['CelestialBody']):
    """
    During each iteration of `solve_ivp` the semi-major-axes and spin-frequencies are updated dynamically.
    This function resets their values to those specified on creation by the function `create_submoon_system`.
    :param y: list,                                A hard copy of the values assigned to the `planetary_system`
                                                    on creation.
                                                    (Copies to avoid using the dynamically updated values).
    :param planetary_system: List['CelestialBody'], The planetary system to reset.
    """
    # Unpack all values to update
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y

    star, planet, moon, submoon = planetary_system

    star.update_spin_frequency_omega(omega_s)
    planet.update_spin_frequency_omega(omega_p)
    moon.update_spin_frequency_omega(omega_p)

    planet.update_semi_major_axis_a(a_s_p)
    moon.update_semi_major_axis_a(a_p_m)
    submoon.update_semi_major_axis_a(a_m_sm)


def check_if_stiff(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    """
    If the number of oscillations at any interval length (sign change in the
    derivative of any variable) is large relative to the time interval, the problem is likely to be stiff.
    In that case, skip this iteration.
    This assumes that stiff system will always change sign from one step directly to the next, e.g.
    calculate_sign_changes([1, 2, 3, -1, -1]) will return 1, calculate_sign_changes([-1, 2, -3, 1, -1]) four.
    """
    dy_dt = submoon_system_derivative(t, y, planetary_system, mu_m_sm, mu_p_m, mu_s_p)
    # Update tracker
    tracker.add(y, dy_dt, t)
    # NOTE : The array tracker.t must not be exactly equal to the by the solver finally returned time points,
    # which in this case does not have an important effect, I assume
    current_step_index = np.where(tracker.t == t)[0][0]

    # The columns of the tracker.y matrix, i.e. the elements (=rows) of the tracker.y.T matrix represent the time
    # evolution of a single variable
    dy_dt_evolutions = tracker.dy_dt.T
    debug_arr = [False, False, False, False, False, False, False]  # index 3 possible ill
    eta_coeffs = [stiffness_coefficient(derivative_evol, sl=5, debug=debug) for derivative_evol,
                  debug in zip(dy_dt_evolutions, debug_arr)]
    # print("Eta coefficients (a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s): ", eta_coeffs)
    eta = np.max(eta_coeffs)
    tracker.add_eta(eta)
    threshhold = 0.7
    # threshhold = 0.843
    # if eta > threshhold - 0.05:
    #     print("\t\t\tCurrent eta: ", eta, " at time point ", t/(3600 * 24 * 365 * 1e9), " Gyrs.")
    # threshhold = 0.65  # for testing purposes. Or maybe even keep this
    # threshhold = 0.4  # for testing purposes. Or maybe even keep this

    control_plot = True
    if control_plot and tracker.eta_chain[current_step_index] > threshhold and tracker.iter_counter == (3,4,7):
        time = tracker.t
        vals = tracker.y[:, 3]
        fig, ax = plt.subplots()
        ax.plot(time/(3600 * 24 * 365 * 1e9), vals, label=r"$a_{sm}$?", color=red)
        ax2 = ax.twinx()
        ax2.plot(time/(3600 * 24 * 365 * 1e9), np.gradient(vals, time), label=r"$a_{sm}$ grad")
        ax.set_xlabel(r"Time in $\mathrm{Gyr}$")
        ax.legend(loc="lower right")
        ax2.legend(loc="upper right")
        plt.show()

    return tracker.eta_chain[current_step_index] - threshhold


def stiffness_coefficient(arr, sl=5, tolerance_percent=68, debug=False):
    """
    Calculates the stiffness coefficient for a given array.
    Divides `arr`, for which the gradient is to be passed, into sub-arrays of length 5.
    Converts the numbers of those arrays to either -1 or +1 based on the sign of the derivative.
    Sums up the values of the sub-array.

    Parameters:
    - arr: A 1D numpy array of numerical values.
    - subarray_length: Length of each subarray to divide the array into.
    - tolerance_percent: Percentage of subarray length to define closeness to 0 (default is 68%).

    Returns:
    - A float representing the fraction of subarrays with a summed value close to 0, meaning that the values of the
    array average out on small scales => Small scale oscillations.
    """
    subarray_length = sl
    arr = np.array(arr)
    total_length = len(arr)
    how_much_larger = 10  # we want the array to be ten times larger than the sub_array a length for a proper inference

    if subarray_length <= 0 or total_length < how_much_larger * subarray_length:
        # The array is too small to accurately infer the stiffness, return a small number so stiffness detector is not
        # triggered (triggered at 0.8)
        return 0.01
        # raise ValueError("Subarray length must be positive and less than or equal to the length of the array.")

    if subarray_length < 5:
        raise ValueError("Each subarray must be at least length 5.")

    # continue, but cut-off the first (how_much_larger*subarray_length) values to only include the potentially
    # oscillating regime
    # c = int(np.round((how_much_larger*subarray_length/2)))  # cut of the first how_much_larger/2 values!
    # arr = arr[c:]
    subarray_sums = []

    for i in range(0, total_length - subarray_length + 1, subarray_length):
        subarray = arr[i:i + subarray_length]
        converted = np.where(subarray < 0, -1, 1)
        subarray_sum = np.sum(converted)
        subarray_sums.append(subarray_sum)

    subarray_sums = np.array(subarray_sums)
    tolerance = (tolerance_percent / 100) * subarray_length

    # np.abs(subarray_sums) <= tolerance does: for each el in subarray_sums, check if abs(el) <= tolerance, if yes
    # return 1, else 0. Default tolerance 3.4
    oscillations = (np.abs(subarray_sums) <= tolerance)  # Array of [False, True] etc that indicates whether in each
    # subarray oscillations were detected or not

    # One final operation: Cut out possible non-oscillating regime at the start which is not of interest.
    # This is only to cut computational cost for large arrays
    if not oscillations[0] and len(oscillations) > 30 and np.any(oscillations):
        index_of_first_true = np.where(oscillations)[0][0]
        oscillations = oscillations[index_of_first_true:]

    # Final final edit: If array is to big, just please look at the last 20 elements
    if len(oscillations) > 30:
        oscillations = oscillations[-20:]

    if debug:
        print("Oscillations: ", oscillations, " -> summed: ", np.sum(oscillations), " which then is diviedd by ",
              len(oscillations))
    close_to_zero_fraction = np.sum(oscillations) / len(oscillations)

    return close_to_zero_fraction


def calculate_sign_changes(arr):
    """
    Counts the number of times the sign changes in the given array.

    Parameters:
    - arr: A 1D numpy array of numerical values.

    Returns:
    - An integer representing the number of sign changes in the array.

    Examples:
     >> calculate_sign_changes([])
    0
    >> calculate_sign_changes([1, 2, 3, -1, -1])
    1
     >> calculate_sign_changes([-1, 2, -3, 1, -1])
    4
    """
    if len(arr) == 0:
        return None
    sign_changes = np.diff(np.sign(arr))
    return np.sum(sign_changes != 0)


def update_values(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    """
    Note
    ----
    Update semi-major-axes and spin-frequencies via class method. This is to be passed to the `events` argument
    of `solve_ivp`.
    Check if the Roche-Limit or Hill-Radius of the bodies is trespassed based on the updated semi-major-axis values.
    If yes, terminate the program (terminal attribute set to True).
    """
    # track and update all relevant values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y

    # Calculate orbit frequencies
    n_m_sm = keplers_law_n_from_a_simple(a_m_sm, mu_m_sm)
    n_p_m = keplers_law_n_from_a_simple(a_p_m, mu_p_m)
    n_s_p = keplers_law_n_from_a_simple(a_s_p, mu_s_p)

    # Unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system

    star.update_spin_frequency_omega(omega_s)
    planet.update_spin_frequency_omega(omega_p)
    moon.update_spin_frequency_omega(omega_p)

    planet.update_semi_major_axis_a(a_s_p)
    moon.update_semi_major_axis_a(a_p_m)
    submoon.update_semi_major_axis_a(a_m_sm)

    # Before returning, document if any sign changes occurred in the derivatives
    tracker.add_signs((c_sign(omega_m - n_m_sm), c_sign(omega_p - n_p_m), c_sign(omega_s - n_s_p)))

    # Use an infinity value, so to not actually activate the event
    return omega_s - np.inf


def track_sm_m_axis_1(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Track: Will the submoon fall under the moon's roche limit?
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    rl = submoon.get_current_roche_limit()
    # if a_m_sm - rl < 0:
    #    print("\t\tThe submoon's semi-major-axis fell under the moon's roche limit.")
    #    print("val of a_sm: ", a_m_sm, " roche limit: ", rl)
    return a_m_sm - rl

def track_sm_m_axis_2(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):

    # Track: Will the submoon escape the moon's influence?
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    crit_a = submoon.get_current_critical_sm_axis()
    # if a_m_sm - crit_a > 0:
    #    print("\t\tThe submoon's semi-major-axis surpassed the critial semi-major-axis. Val of current axis:", a_m_sm,
    #          " and val of current crit a ", crit_a)
    return a_m_sm - crit_a


def track_m_p_axis_1(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Track: Will the moon fall under the planet's roche limit?
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    rl = moon.get_current_roche_limit()
    # if a_p_m - rl < 0:
    #     print("\t\tThe moon's semi-major-axis fell under the planet's roche limit.")
    return a_p_m - rl


def track_m_p_axis_2(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Track: Will the moon escape the planet's influence?
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    crit_a = moon.get_current_critical_sm_axis()
    # if a_p_m - crit_a > 0:
    #     print("\t\tThe moons's semi-major-axis surpassed the critical semi-major-axis.")
    return a_p_m - crit_a

update_values.terminal = False
track_sm_m_axis_1.terminal = True
track_sm_m_axis_2.terminal = True
track_m_p_axis_1.terminal = True
track_m_p_axis_2.terminal = True
check_if_stiff.terminal = True
check_if_stiff.direction = 1


def solve_ivp_iterator_console_logger(planetary_system, mode=0, current_y_init=None, upper_lim_out=None,
                                      upper_lim_middle=None, upper_lim_in=None, outer_counter=None,
                                      middle_counter=None, inner_counter=None):
    sun_mass = 1.98847 * 1e30  # kg
    earth_mass = 5.972 * 1e24  # kg
    luna_mass = 0.07346 * 1e24  # kg
    jupiter_mass = 1.89813 * 1e27  # kg

    AU = 149597870700  # m
    earth_luna_distance = 384400000  # m

    star, planet, moon, submoon = planetary_system

    star_mass_relative = np.round(star.mass / sun_mass, 4)
    planet_mass_in_earths = np.round(planet.mass / earth_mass, 4)
    planet_mass_in_jupiters = np.round(planet.mass / jupiter_mass, 4)
    planet_mass_relative = planet_mass_in_earths if planet_mass_in_earths < 200 else planet_mass_in_jupiters
    planet_mass_reference_str = "m_earth" if planet_mass_in_earths < 200 else "m_jup"
    moon_mass_relative = np.round(moon.mass / luna_mass, 4)
    submoon_mass_relative = np.round(submoon.mass / moon.mass, 4)

    planet_distance_relative = np.round(planet.a / AU, 4)
    moon_distance_relative = np.round(moon.a / earth_luna_distance, 4)
    submoon_distance_relative = np.round(submoon.a / earth_luna_distance, 4)

    if mode == 0:
        # General information about the masses that can be printed once in the beginning
        print("\n------------------------------------------")
        print("The following base system is simulated:")
        print(f"Sun: {star_mass_relative} m_sun → Planet: {planet_mass_relative} {planet_mass_reference_str} → "
              f"Moon: {moon_mass_relative} m_luna → Submoon: {submoon_mass_relative} m_moon.\n")

    elif mode == 1:
        # More detailed information about the initial values of the system describing the objects,
        # that change in each iteration.
        if any((current_y_init, upper_lim_out, upper_lim_middle, upper_lim_in,
                outer_counter, middle_counter, inner_counter)) is None:
            raise ValueError("\t\tLogging-Mode-1 arguments must not be None.")

        print(f"\t\tSub-iteration {outer_counter, middle_counter, inner_counter}. "
              f"Initial distances between objects in this iteration:")

        i_index = 2  # Planet position in `y_init`
        j_index = 1  # Moon position in `y_init`
        k_index = 0  # Submoon position in `y_init`

        submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0 = current_y_init
        relative_distances = [submoon_distance_relative, moon_distance_relative, planet_distance_relative]

        current_iteration_value_i = current_y_init[i_index]
        current_iteration_value_j = current_y_init[j_index]
        current_iteration_value_k = current_y_init[k_index]

        progress_in_percent_planet = 100 * np.round(current_iteration_value_i / upper_lim_out, 2)
        progress_in_percent_moon = 100 * np.round(current_iteration_value_j / upper_lim_middle, 2)
        progress_in_percent_submoon = 100 * np.round(current_iteration_value_k / upper_lim_in, 2)

        relative_distances[i_index] = (f'{progress_in_percent_planet}% of '
                                       f'upper limit. Current abs. value: {np.round(
                                           current_iteration_value_i / AU, 2)}')
        relative_distances[j_index] = (f'{progress_in_percent_moon}% of '
                                       f'upper limit. Current abs. value: {np.round(
                                           current_iteration_value_j / earth_luna_distance, 2)}')
        relative_distances[k_index] = (f'{progress_in_percent_submoon}% of '
                                       f'upper limit. Current abs. value: {np.round(
                                           current_iteration_value_k / earth_luna_distance, 2)}')

        submoon_distance_relative, moon_distance_relative, planet_distance_relative = relative_distances
        print(f"\t\tSun -- {planet_distance_relative}AU --> Planet -- {moon_distance_relative} d_luna --> Moon -- "
              f"{submoon_distance_relative} d_luna --> Submoon\n")


class InitialValuesOutsideOfLimits(ValueError):
    pass


class RocheLimitGreaterThanCriticalSemiMajor(ValueError):
    pass


def sanity_check_initial_values(a_sm_0: float, a_m_0: float, a_p_0: float, planetary_system: List['CelestialBody']):
    """
    Sanity checks if the initial values relevant for termination of the simulation (semi-major-axis of moon, submoon
    and planet) don't, from the get-go, surpass or fall under their respective a_crit or roche-limit.
    """
    star, planet, moon, submoon = planetary_system

    r_p_l = planet.get_current_roche_limit()
    # a_p_crit = planet.get_current_critical_sm_axis() This will throw an error, try it

    r_m_l = moon.get_current_roche_limit()
    a_m_crit = moon.get_current_critical_sm_axis()

    r_sm_l = submoon.get_current_roche_limit()
    a_sm_crit = submoon.get_current_critical_sm_axis()

    # Order: submoon, moon, planet
    list_a_crit = [a_sm_0 / a_sm_crit, a_m_0 / a_m_crit,
                   .5]  # Checking that initial sm.axis are not already above critical
    # sm. axis. The .5 was: a_p_0/a_p_crit. See comment at a_p_crit.
    list_r_lim = [r_sm_l / a_sm_0, r_m_l / a_m_0,
                  r_p_l / a_p_0]  # Checking that initial sm.axis are not already under roche
    # limit
    list_a_crit_r_lim_cross = [r_sm_l / a_sm_crit, r_m_l / a_m_crit, .5]  # Checking that critical semi-major-axes are
    # bigger than roche limits
    names = ["submoon", "moon", "planet"]

    for index, val in enumerate(list_a_crit_r_lim_cross):
        if val > 1:
            raise RocheLimitGreaterThanCriticalSemiMajor(
                f'A priori ratio of roche limit of {names[index]} and initial critical s.m.axis '
                f'bigger than one: {val}')

    # Raise value error if any of those conditions becomes bigger than 1
    for index, val in enumerate(list_a_crit):
        if val > 1:
            raise InitialValuesOutsideOfLimits(
                f'A priori ratio of initial semi-major-axis of {names[index]} and its critical s.m.axis '
                f'bigger than one: {val}')

    for index, val in enumerate(list_r_lim):
        if val > 1:
            raise InitialValuesOutsideOfLimits(
                f'\t\tA priori ratio of initial semi-major-axis of {names[index]} and its parent bodys '
                f'roche limit bigger than one: {val}')



def premature_termination_logger(er):
    print("\t\tStability status: INIT VAL ER")
    print("\t\tTermination reason: Bad initial value parameters.")
    print(f"\t\tInitial values are not within boundary conditions (e.g., a_crit < roche_limit a priori). Exception:"
          f"\n\t\t{er}")
    print("\t\tSkipping iteration.\n")
    print("\n------------------------------------------\n")


def log_results(status, termination_reason, time_points):
    if status == 0:
        # Stable input parameters.
        stability_status = "+1"
    elif status == 1:
        # A termination event occurred
        stability_status = "-1"
    elif status == -1:
        # Numerical error.
        stability_status = "NUMERICAL ER"
    else:
        raise ValueError("Unexpected outcome.")

    print(f"\t\tStability status: {stability_status}")
    print("\t\tTermination reason: ", termination_reason)
    print(f"\t\tNum of time steps taken: {len(time_points)}, age reached: "
          f"{turn_seconds_to_years(time_points[-1])} years.")
    print("\t\tGoing to next iteration.\n")
    print("\n------------------------------------------\n\n")

    return stability_status


def plt_results(sol_object, planetary_system, list_of_mus, print_sol_object=False, save=False, show=True,
                old_state=None, plot_derivatives=False):
    # Unpack variables outputted by solution object.
    (time_points, solution, t_events, y_events, num_of_eval, num_of_eval_jac, num_of_lu_decompositions, status,
     message, success) = unpack_solve_ivp_object(sol_object)
    if print_sol_object:
        print("sol_object : ", sol_object)

    state_vector_plot(time_points, solution, submoon_system_derivative, planetary_system, list_of_mus, show=show,
                      save=save, plot_derivatives=plot_derivatives, old_state=old_state)


def find_termination_reason(status, t_events, keys):
    if status == 1:
        term_index = [i for i, array in enumerate(t_events) if bool(len(array))]
        if len(term_index) > 1:
            raise ValueError("More than one termination event")
        return keys[term_index[0]]
    else:
        return "No termination event occurred."


def document_result(status, termination_reason, time_points, results, i, j, k,
                    y_init, y_final, termination_reason_counter, lifetimes, whole_solution_object,
                    interesting_events_dict):

    stability_status = log_results(status, termination_reason, time_points)  # Prints in terminal

    results.append(stability_status)  # Appends to object that is returned at the end of grid search

    lifetimes.append([(i, j, k,), turn_seconds_to_years(np.max(time_points)), y_init, y_final,
                      termination_reason, whole_solution_object, tracker.get_sign_changes()])

    termination_reason_counter[str(termination_reason)] += 1

    if str(termination_reason) != "No termination event occurred.":
        interesting_events_dict[str(termination_reason)].extend([(i,j,k)])


def pickle_me_this(filename: str, data_to_pickle: object):
    path = filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()


def unpickle_me_this(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def consistency_check_volume(expected_volume, actual_volume):
    if actual_volume != expected_volume:
        raise_warning(
            f"\nThere is a mismatch between the expected volume of the grid search (n_pl * n_moon * n_submoon) "
            f"\n({expected_volume}) and the actual volume ({actual_volume}), which means some cases"
            f"\nhave been overlooked. This is not detrimental, but leads to the solution cube being lower res "
            f"than necessary.")


def print_final_data_object_information(hits, computation_time, termination_reason_counter, lifetimes, norm):
    """

    :param hits:                        An array that indicates success (+1) or failure (-1) of iteration.
    :param computation_time:            Computation time in seconds.
    :param termination_reason_counter:  A dict containing the termination reasons.
    :param lifetimes:                   The `lifetimes` object returned by the DFE iterative solver.
    :param norm:                        The number of years that are interpreted as stable.
    :return:
    """
    from collections import defaultdict

    # Turn  `hits` array into `hits` dict to print more easily
    hit_counts = defaultdict(int)
    for hit in hits:
        hit_counts[hit] += 1

    lts = np.array([lifetime[1] for lifetime in lifetimes]) / norm
    stiffs = termination_reason_counter["Iteration is likely to be stiff"]
    quo = np.round(100 * np.count_nonzero(lts == 1) / (len(lts) - stiffs), 2)

    print("\n\n--------RESULTS--------\n")

    print("\nTotal number of iterations: ", len(hits), ".")
    print(f"Of those that were non-stiff ({len(lts) - stiffs}), ", quo, "% reached the stable lifetime of ", norm,
          " years.")
    print(f"{100*stiffs/len(lts)} % iterations were not finished due to stiffness.\n")
    print("\n------ Stability states (-1: Termination, NUM ER: Numerical error, +1: Longevity):")
    for key, value in hit_counts.items():
        print(f'{key}: {value}')
    print("------")

    print("\n------ Total computation time in minutes: \n", computation_time / 60)
    print("------")

    print("\n------ Histogram of termination reasons:")
    for key, value in termination_reason_counter.items():
        print(f'{key}: {value}')
    print("------")

    print("\n------ First five elements of iteration with longest lifetime (Second element is lifetime in years):")
    max_subarray = max(lifetimes, key=lambda x: x[1])
    index_of_longest_lifetime_array = lifetimes.index(max_subarray)
    print(lifetimes[index_of_longest_lifetime_array][:5])


def showcase_results(data, plot_initial_states=False, suppress_text=True, plot_final_states=True, save=False, filename=None):
    """

    :param data, array:      The unpickled data returned by the function differential equation solver.
    :param plot_initial_states, bool:         Whether or not to plot a 3D result cube of initial states.
    :param plot_final_states, bool:         Whether or not to plot a 3D result cube of final states.
    :param suppress_text, bool Whether or not to print basic information about the grid search.

    For convenience, the structure of one element in the `lifetimes` object:

            [0]: (i, j, k,)
            [1]: turn_seconds_to_years(np.max(time_points))
            [2]: y_init
            [3]: y_final
            [4]: termination_reason
            [5]: whole_solution_object
            [6]: tracker.get_sign_changes()

    :return:
    """
    # Unpack data object
    hits, termination_reason_counter, lifetimes, physically_varied_ranges, computation_time = data

    # Normalization constant to determine transparency value (0-1) of a given 3D datapoint.
    # 1 if lifetime == normalization, lifetime/normalization else
    normalization = 4.5e9  # years

    sign_changes_arr = [lt[6] for lt in lifetimes if (lt[6] != [None, None, None] and lt[6] != [0, 0, 0])]
    print("Non-Trivial sign changes throughout: ", sign_changes_arr)

    if not suppress_text:
        # Print basic information about grid search
        print_final_data_object_information(hits, computation_time, termination_reason_counter, lifetimes,
                                            norm=normalization)

    # Get the varied arrays (X, Y, Z) of grid search, physical as well as integers representing the iteration count
    varied_planet_range, varied_moon_range, varied_submoon_range = tuple(
        [physically_varied_ranges[i] for i in range(3)])
    n_pix_planet, n_pix_moon, n_pix_submoon = tuple([len(physically_varied_ranges[i]) for i in range(3)])

    # Sanity-check grid search volume
    consistency_check_volume(expected_volume=n_pix_planet * n_pix_moon * n_pix_submoon, actual_volume=len(lifetimes))

    solved_coordinates = np.array([lifetime[0] for lifetime in lifetimes if
                                   lifetime[4] != "Iteration is likely to be stiff"])
    unsolved_coordinates = np.array([lifetime[0] for lifetime in lifetimes if
                                     lifetime[4] == "Iteration is likely to be stiff"])
    # `lifetime[0]` is iteration counter (i, j, k,) (proxies for actual physical values).
    # I.e. `coordinates` is now a nx3 matrix where each row represents a
    # coordinate point: [[ 1  1  1], [ 1  1  2], [ 1  1  3] ... ].
    # The difference between `solved_coordinates` and `unsolved_coordinates` is that `solved_coordinates` does NOT
    # contain those iterations in which the stiffness ratio indicated that the numerics are likely to be stiff and
    # the iteration was therefore prematurely interrupted.
    # These iterations are collected in `unsolved_coordinates` which are to be marked with an `x` in the 3D plot.

    X_solved = [coordinate_triple[0] for coordinate_triple in solved_coordinates]  # Planet semi-major-axis proxy
    Y_solved = [coordinate_triple[1] for coordinate_triple in solved_coordinates]  # Moon -""-
    Z_solved = [coordinate_triple[2] for coordinate_triple in solved_coordinates]  # Submoon -""-
    # These operations are equivalent to extracting the first, second and third column of `coordinates` respectively.

    X_unsolved = [coordinate_triple[0] for coordinate_triple in unsolved_coordinates]
    Y_unsolved = [coordinate_triple[1] for coordinate_triple in unsolved_coordinates]
    Z_unsolved = [coordinate_triple[2] for coordinate_triple in unsolved_coordinates]

    X_solved = np.array([varied_planet_range[coordinate_proxy - 1] for coordinate_proxy in X_solved]) / AU
    # Planet semi-major-axis physical in AU
    Y_solved = np.array([varied_moon_range[coordinate_proxy - 1] for coordinate_proxy in Y_solved]) / LU  # Moon -""-
    # in earth-moon distances
    Z_solved = np.array([varied_submoon_range[coordinate_proxy - 1] for coordinate_proxy in Z_solved]) / SLU  # Submoon
    # -""- in twentieths of earth-moon distance

    X_unsolved = np.array([varied_planet_range[coordinate_proxy - 1] for coordinate_proxy in X_unsolved]) / AU
    Y_unsolved = np.array([varied_moon_range[coordinate_proxy - 1] for coordinate_proxy in Y_unsolved]) / LU
    Z_unsolved = np.array([varied_submoon_range[coordinate_proxy - 1] for coordinate_proxy in Z_unsolved]) / SLU

    values_solved = np.array([lifetime[1] for lifetime in lifetimes
                              if lifetime[4] != "Iteration is likely to be stiff"]) / normalization

    values_unsolved = np.array([lifetime[1] for lifetime in lifetimes
                                if lifetime[4] == "Iteration is likely to be stiff"]) / normalization
    # a number between 0 and 1, if 1, equates to a point in phase space that has a stable lifetime of `normalization`
    # years.

    # Now, finally, get the distribution of final states. These here also include the end states of stiff equations.
    X_solved_f = []
    Y_solved_f = []
    Z_solved_f = []
    values_solved_f = []

    for lt in lifetimes:
        if lt[3] is not None:
            # exclude premature interrupted iterations (but include stiff ones)
            end_state = lt[3][:3]
            a_sm_f, a_m_f, a_p_f = end_state
            X_solved_f.append(a_p_f)
            Y_solved_f.append(a_m_f)
            Z_solved_f.append(a_sm_f)
            values_solved_f.append(lt[1]/normalization)

    X_solved_f = np.array(X_solved_f) / AU
    Y_solved_f = np.array(Y_solved_f) / LU
    Z_solved_f = np.array(Z_solved_f) / SLU

    # A small snippet to export data to embed onto my personal webpage using chartJS.
    # Do not necessarily delete
    """
    export_data_for_example = False
    if export_data_for_example:
        # stability transition region, this is in order to post an example unto `https://iason-saganas.github.io`
        transition_indices = np.where((values >= 0.75) & (values <=0.9))

        copy_xyz_data_to_clipboard = False
        if copy_xyz_data_to_clipboard:
            import pyperclip
            # Convert array to string with elements separated by commas
            # array_str = '\n'.join(['[' + ', '.join(map(str, row)) + '],' for row in coordinates])
            values_to_copy = list(zip(X[transition_indices], Y[transition_indices], Z[transition_indices]))
            array_str = '\n'.join(['[' + ', '.join(map(str, coordinate_tuple)) + '],' for coordinate_tuple in values_to_copy])

            # Copy the string to the clipboard
            pyperclip.copy(array_str)
    """

    plot_3d_voxels_initial_states(data_solved=[X_solved, Y_solved, Z_solved, values_solved],
                                  data_unsolved=[X_unsolved, Y_unsolved, Z_unsolved, values_unsolved],
                                  save=save, filename=f"initial_states_{filename}", plot=plot_initial_states)
    plot_3d_voxels_final_states(data=[X_solved_f, Y_solved_f, Z_solved_f, values_solved_f],
                                save=save, filename=f"end_states_{filename}", plot=plot_final_states)


def jacobian(t, y, *args):
    """
    The calculations of the Jacobian were done by hand.
    Note how these calculations inherently assume ∂_x sgn(x) = 0, which is wrong, but should be a good approximation
    if synchronicity is never achieved.
    But, since synchronicity is not something we want to forbid a priori, I suggest to not use this Jacobian.
    Although finite difference approximation reproduces quite well what this jacobian does.
    :return:
    """
    # print("Current time step in Myrs: ", t/(3600*24*365*1e6))
    jac = np.zeros((6, 6))

    planetary_system, mu_m_sm, mu_p_m, mu_s_p = args
    star, planet, moon, submoon = planetary_system

    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y

    n_m_sm = keplers_law_n_from_a_simple(a_m_sm, mu_m_sm)
    n_p_m = keplers_law_n_from_a_simple(a_p_m, mu_p_m)
    n_s_p = keplers_law_n_from_a_simple(a_s_p, mu_s_p)

    eq1 = get_a_factors(hosted_body=submoon) * np.sign(omega_m - n_m_sm) * (-11 / 2) * a_m_sm ** (-13 / 2)
    eq2 = get_a_factors(hosted_body=moon) * np.sign(omega_p - n_p_m) * (-11 / 2) * a_p_m ** (-13 / 2)
    eq3 = get_a_factors(hosted_body=planet) * np.sign(omega_s - n_s_p) * (-11 / 2) * a_s_p ** (-13 / 2)
    eq4 = -get_omega_factors(body=moon) * np.sign(omega_m - n_m_sm) * submoon.mass ** 2 * (-6) * a_m_sm ** (-7)
    eq5 = -get_omega_factors(body=moon) * np.sign(omega_m - n_p_m) * planet.mass ** 2 * (-6) * a_p_m ** (-7)
    eq6 = -get_omega_factors(body=planet) * np.sign(omega_p - n_p_m) * moon.mass ** 2 * (-6) * a_p_m ** (-7)
    eq7 = -get_omega_factors(body=planet) * np.sign(omega_p - n_s_p) * star.mass ** 2 * (-6) * a_s_p ** (-7)
    eq8 = -get_omega_factors(body=star) * np.sign(omega_s - n_s_p) * planet.mass ** 2 * (-6) * a_s_p ** (-7)

    jac[0, 0] = eq1;
    jac[1, 1] = eq2;
    jac[2, 2] = eq3;
    jac[3, 0] = eq4;
    jac[3, 1] = eq5;
    jac[4, 1] = eq6
    jac[4, 2] = eq7;
    jac[5, 2] = eq6
    return jac


def calculate_angular_momentum(planetary_system, mu_m_sm, mu_p_m, mu_s_p, y_state_vector_evolution):
    """
    :param planetary_system:
    :param mu_m_sm:
    :param mu_p_m:
    :param mu_s_p:
    :param y_state_vector_evolution: must be sol.T
    :return:
    """

    total_angular_momentum_evolution = []
    for column in y_state_vector_evolution:
        star, planet, moon, submoon = planetary_system
        a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = column

        n_m_sm = keplers_law_n_from_a_simple(a_m_sm, mu_m_sm)
        n_p_m = keplers_law_n_from_a_simple(a_p_m, mu_p_m)
        n_s_p = keplers_law_n_from_a_simple(a_s_p, mu_s_p)

        L_spin = omega_m * moon.I + omega_p * planet.I + omega_s * star.I  # Σ_i I_i \dot{θ}
        L_orbit = submoon.mass * n_m_sm * a_m_sm ** 2 + moon.mass * n_p_m * a_p_m ** 2 + planet.mass * n_s_p * a_s_p ** 2
        # Σ_i m_i \dot{φ} a^2
        total_angular_momentum_evolution.append(L_spin + L_orbit)
    return np.array(total_angular_momentum_evolution)


def sanity_check_tot_ang_momentum_evolution(planetary_system, mu_m_sm, mu_p_m, mu_s_p, y_state_vector_evolution):
    """
    If the values of the array that is returned by `calculate_angular_momentum` differ significantly from eachother,
    this might be an artifact bad handling of stiffness in case an explicit solver has been used.
    The explicit solver does not resolve small scale oscillatory structure (compare `RK45` vs. `Radau` method), but
    the solution is taken to be satisfactory as long as angular momentum is conserved throughout the evolution, since
    we are only interested in whether or not the system will remain stable, not the exact values of the semi-major-axes.

    y_state_vector_evolution must be solution.T

    :return:
    """
    tot_ang_mom = calculate_angular_momentum(planetary_system, mu_m_sm, mu_p_m, mu_s_p, y_state_vector_evolution)
    tot_ang_mom0 = tot_ang_mom[0]
    residuals = np.abs(tot_ang_mom - tot_ang_mom0)
    threshhold = 0.01 * tot_ang_mom0
    if np.any(residuals > threshhold):
        raise ValueError("It seems like total angular momentum was not conserved within 1% of the starting angular "
                         "momentum during this run. I suggest to rerun"
                         "using an implicit solver method.")
    else:
        print("\t\tAngular momentum is ... conserved within 1% of starting angular momentum.")


def sanity_check_with_analytical_sol_if_possible(planetary_system, mu_m_sm, mu_p_m, mu_s_p, y_state_vector_evolution,
                                                 time_points):
    """

    The workflow-of this function is as follows:

    Check whether comparison to the analytical solution is applicable (only if no sign changes
    occurred in the semi-major-axes evolutions during the numeric integration).
    If not, return, if yes continue.

    Are the numeric end-states near to the analytical solution? If not:

    Does the analytic solution show signs of divergence?
    --------------------------------------------------------
    Check by looking for nan's in the analytic solution and get the time point of the nan ~ time point of divergence.

       No
       => Analytic and numerical final state have to be within 1%.

       Yes
       => Has the numeric solution diverged?
       ----------------------------------------

       Check by computing relative time differences.
       If progress was made only 1/1000th fraction in the time direction, this indicates extremely small time-steps,
       which we is a sign of divergence in case there are no rapid oscillations in the solution (which they are not
       at this step in the code-workflow, they would have already been caught.)

            No
            => Raise value error: Significant deviation from analytical solution.

            Yes
            => Have the solutions diverged in the same direction?
            (Check sign of gradients)
            --------------------------------------------------------

                No
                => raise value error: Significant deviation from analytical solution.

                Yes
                => Are the time points of the analytic and numerical solution not more than a 1000yrs apart?
                -----------------------------------------------------------------------------------------------

                    No
                    => raise value error: Significant deviation from analytical solution.

                    Yes
                    => ACCEPT


    If the solution is accepted it means that either the numerical solution follows the analytical one, or that
    both have diverged, in the same direction, not more than a 1000yrs apart. This offset in the time direction
    is then explained as the consequence of accumulated round-off errors in the numeric solution since the solver
    does not jump soon enough abruptly to the necessary, extremely small time-step, but increases it gradually.

    There is one case, that this workflow does not catch.

    --


    :param planetary_system:            The planetary system.
    :param mu_m_sm:                     The standard gravitational parameter between moon and submoon.
    :param mu_p_m:                      The standard gravitational parameter between moon and planet.
    :param mu_s_p:                      The standard gravitational parameter between planet and star.
    :param y_state_vector_evolution:    The full time evolution of the state vector from `solve_ivp`, transposed.
    :param time_points:                 The time points at which the numeric solution was calculated
    :return:
    """

    # First check if the analytic solution is applicable (only if no sign changes occurred in the semi-major-axes
    # during the numeric integration)

    final_time = time_points[-1]
    sign_counters = tracker.get_sign_changes()
    if sign_counters != [0, 0, 0]:
        print("\t\tAnalytic comparison not applicable, sign changes in (a_sm, a_m, a_p): ", sign_counters)
        # print("\n------------------------------------------\n")
        return

    # sign_counters == [0, 0, 0], therefore Comparison applicable

    list_of_mus = [mu_m_sm, mu_p_m, mu_s_p]
    initial_state = y_state_vector_evolution[0]
    end_state = y_state_vector_evolution[-1]  # rows: state at different times, columns: variables

    # Has any of the analytic solutions a_sm, a_m or a_p diverged?
    extended_time = np.sort(np.concatenate((np.linspace(max(time_points)*0.9, max(time_points)*1.2, 1000), time_points)))
    ana_evols = np.array(semi_major_axes_analytical_solution(time_points, initial_state,
                                                             planetary_system, list_of_mus))
    ana_evols_ext = np.array(semi_major_axes_analytical_solution(extended_time, initial_state,
                                                                 planetary_system, list_of_mus))
    A = ana_evols_ext.T  # Shorter Alias, the transpose is for the `first_nan_indices` line
    first_nan_indices = [np.where(np.isnan(A[:, col]))[0][0] if np.any(np.isnan(A[:, col])) else None for col in
                         range(A.shape[1])]
    ana_is_divergent = [True if idx is not None else False for idx in first_nan_indices]

    a_m_sm_NUM_f, a_p_m_NUM_f, a_s_p_NUM_f, _, _, _ = end_state  # at numerical time points
    a_m_sm_ANA_f, a_p_m_ANA_f, a_s_p_ANA_f = [a[-1] for a in ana_evols]  # also at numerical time points!

    numeric_end_state = [a_m_sm_NUM_f, a_p_m_NUM_f, a_s_p_NUM_f]
    analytic_end_state = [a_m_sm_ANA_f, a_p_m_ANA_f, a_s_p_ANA_f]

    names = ['submoon', 'moon', 'planet']

    # Before pushing down further in the pipeline, do a preliminary check of whether the final states are near to
    # each other, in which case further analysis is not needed.

    is_near = [False, False, False]
    for idx, body in enumerate(names):
        # Possibly good solutions
        is_near[idx] = handle_non_diverged_solution(numeric_end_state[idx], analytic_end_state[idx], sign_counters,
                                                    body, preliminary=True)

    for idx, analytical_sub_solution_is_diverged in enumerate(ana_is_divergent):
        # Possibly bad solutions
        if is_near[idx]:
            # If the solution has previously tested to be good, no need to do all this for loop.
            continue
        num_sol = y_state_vector_evolution[:, idx]
        ana_sol = ana_evols[idx]  # on numerical time points domain (although does not matter if on extended domain)
        body = names[idx]

        if not analytical_sub_solution_is_diverged:
            # Check for each solution independently!
            handle_non_diverged_solution(numeric_end_state[idx], analytic_end_state[idx], sign_counters, body)

        elif analytical_sub_solution_is_diverged:
            # Case one: The numeric solution has not diverged => Throw error
            # Case two: The numeric solution has diverged => Check time distance to analytic divergence
            time_of_ana_divergence = extended_time[first_nan_indices[idx]-1]  # take the time before the first NaN
            numeric_solution_has_diverged = detect_divergence_2(t_arr=time_points, y_arr=num_sol)
            if numeric_solution_has_diverged:
                n = len(time_points)
                handle_diverged_solution(num_sol, ana_sol, time_of_ana_divergence, final_time, body, num_of_steps=n)
            else:
                print(f"CASE 2 ERROR: The analytic solution of the {body} s.m.axis has blown up, but the \n"
                      f"numeric solution has not. This can mean: 2.1 The numeric divergence detection algorithm is \n"
                      f"not good enough. 2.2 The numerical solution is wrong.")
                raise ValueError("Case 2. The numeric solution is too far off from the analytic solution.")
        else:
            raise ValueError("Unknown case.")
    # print("\n------------------------------------------\n")  # End


def calculate_relative_neighboring_diffs(arr):
    # Outputs an array of length len(arr)-1, whose i-th element is (arr[i+1]-arr[i])/arr[i+1].
    # E.g.: [1 1 2] will output [0 1] and [1 1 2] will output [0 -0.5]
    return np.diff(arr) / arr[:-1]


# Global testing variables. Here i am trying to see whether the delta_t's in `handle_diverged_solution` grow with
# total time or not.
# If they do, this is a hint of some kind of systematic, which I should further think about.
t_arr_final = []  # the numeric final times of different iterations
delta_t_arr_div = []  # The time differences between the divergences of numeric and analytic solution for each of those iters.
num_steps_arr = []


def handle_diverged_solution(num_sol, ana_sol, time_of_ana_divergence, final_numeric_time, body, num_of_steps=None):
    """
    Checks whether or not the numeric and analytic solution have diverged into the same direction (upwards or downwards)
    by comparing gradients and if yes, check that the point of divergences are close enough (e.g. <1000yrs) for the
    discrepancy to be explicable by increased round-off errors of the numerical solution due to increased steps and
    smaller step sizes.

    :param final_numeric_time:      The final numeric time as an approximate time to the divergence time point of the
                                    numeric solution.
    :param body:                    A string that specifies which body's semi-major-axis is under investigation.
    :param time_of_ana_divergence:  The approximate time at which the analytical solution diverges.
    :param num_sol:         The numeric solution of a quantity.
    :param ana_sol:         It's analytical counterpart.
    :param num_of_steps:    The number of steps performed by the numeric solution, for plotting/debugging purposes.
    :return:
    """
    grad_num_sign = np.sign(np.gradient(num_sol)[-1])
    grad_ana_sign = np.sign(np.gradient(ana_sol)[-1])

    if grad_num_sign != grad_ana_sign:
        print(f"CASE 3 ERROR: The analytic solution and the numeric solution blow up into different directions:\n"
              f"(ana / num) {grad_ana_sign} / {grad_num_sign}")
        raise ValueError("Case 3. The numeric solution is too far off from the analytic solution.")

    # Time difference between points of divergence in years:
    delta_t = np.abs(final_numeric_time - time_of_ana_divergence) / (3600 * 24 * 365)  # in years
    print("Final numeric time in years: ", final_numeric_time)
    print("Analytic time point of divergence: ", time_of_ana_divergence)
    tr = 1e9 * 0.0005  # 0.05% of a billion years = 500.000 yrs.
    print("Differences between points of divergences: ", delta_t)
    if delta_t > tr:
        # Very bad case
        print(f"CASE 4 ERROR: The time points of divergence of the analytic and numeric solution are too far off.\n"
              f"({delta_t} years vs {tr} years). This might mean that the threshhold was too strict, or that the "
              f"hypothesis that \n the time delay between the divergences is caused by increased round-off errors due "
              f"to decreased step-\nsize and increased total steps (in turn due to solver behaviour at divergence) "
              f"is incorrect.")
        raise ValueError("Case 4. The numeric solution is too far off from the analytic solution.")

    if num_of_steps is not None:
        # For plotting purposes
        t_arr_final.append(final_numeric_time)
        delta_t_arr_div.append(delta_t)
        num_steps_arr.append(num_of_steps)

    print(f"\t\tAnalytic comparison applicable for final {body} s.m.axis... "
          f"Analytic solution has diverged, as did the numeric one, in the same direction, {delta_t}yrs (<{tr}) apart."
          f" The solution is accepted.")


def handle_non_diverged_solution(numeric_end_state, analytic_end_state, sign_counters, body, preliminary=False):
    """
    Checks whether the passed numeric states agree with the analytic ones within 1%.
    This check is performed in another function not as a vector operation, but via a for loop, i.e. each element
    of [a_sm, a_m, a_p] possible checked independently.

    This function can be called with an argument `preliminary`, that does not throw an error if the relative
    deviation threshold is surpassed.
    This way, the truth values that are returned (True for is_near analytical and False for not is_near analytical)
    can be grabbed.
    The truth values in turn can be used to decide whether the numerical solution should be pushed further down
    the `sanity_check_with_analytical_sol_if_possible` pipeline.

    :param numeric_end_state:       A list containing the numeric final states of the s.m.-axis of submoon, moon and
                                    planet.
    :param analytic_end_state:      The analytic equivalent to `analytic_end_state`.
    :param sign_counters:           The number of times the sign changed in the derivative of the solutions
                                    (sanity check by eye, when this error is thrown that array must be (0, 0, 0).
    :param body:                    A string that specifies which body's semi-major-axis is under investigation.
    :param preliminary: bool        Whether to throw an error if the relative deviation threshold is surpassed.
    :return:
    """
    relative_deviation = np.array([np.abs(numeric_end_state - analytic_end_state) / analytic_end_state])
    if relative_deviation > 0.01:
        # Numeric solution too far off from analytic one
        if not preliminary:
            print(
                f"\nCASE 1 ERROR:  deviation of numeric solution of {body} s.m.axis (non-diverged analytic end-state) "
                f"too far away from analytic one.\nRelative deviation: {relative_deviation * 100}%. "
                f"But the threshold was 1%. \n",
                f"The analytic solution comparison was applicable since the sign-"
                f"changes in the evolution of (a_sm, a_m, a_p) were: {sign_counters}.")
            raise ValueError("Case 1. The numeric solution is too far off from the analytic solution.")
        return False
    print(f"\t\tAnalytic comparison applicable for final {body} s.m.axis... "
          f"within 1% of expected final analytic value.")
    return True


def detect_divergence(t_arr, y_arr):
    """
    Detects whether y_arr does shows indeed signs of suspected divergence by:
    Assuming that the points in the t_arr get closer and closer together, and finding a transition point after which the
    relative change from one x point to the next is miniscule (e.g. 1e-3 threshhold) and only gets smaller.

    This cuts the t_arr into two regions: A possibly broad region that is suspected to have a relatively small gradient
    and a narrow region with suspected very high gradient.

    λ is defined as the ratio between the gradient in the first region (represented by the gradient sample directly
    before the transition region) and the gradient in the second region (represented by the last gradient sample
    possible, which is taken to be a good characteristic since the second region is suspected to be narrow).

    If λ < 1e-6, return true (diverged), else false (non-diverged)

    :param t_arr:
    :param y_arr:
    :return:
    """
    relative_progress = calculate_relative_neighboring_diffs(t_arr)  # in the t-direction
    print("RELATIVE PROGRESS ARRAY: ", relative_progress)
    try:
        tr_idx = np.where(relative_progress < 1e-3)[0][0]
    except IndexError:
        print("IINDEEX ERRORRR")
        # No transition point was detected.
        return False
    grad = np.gradient(y_arr, t_arr)
    lambda_ratio = grad[tr_idx - 5] / grad[-1]
    print("Calculated lambda ratio: ", lambda_ratio)
    if lambda_ratio < 1e-2:
        return True
    else:
        return False


def detect_divergence_2(t_arr, y_arr):
    """
    An alternative divergence detection method that is to be used in case `t_eval` had been inputted into the solver
    as an `np.linspace`.
    In this case, the support points of the solutions do not get squished more and more together towards the point of
    divergence, which was the fact taken advantage of in `detect_divergence`.

    Now, we may just simply calculate the numerical gradient, calculate the mean gradient at the last three time points
    and the mean gradient at the 25 time points before that, to get

    λ :=  mean_grad_before_last_time_points / mean_grad_last_time_points

    These numbers are just taken heuristically by looking at the numerical gradients of the debug plots, since all
    the divergent curves are essentially self-similar.

    This function should be optimized.
    It's really not that good (see e.g. usage of λ=0.25 as a threshold instead of something like 0.05)

    :return:
    """
    num_grad = np.gradient(y_arr, t_arr)
    mean_grad_end = np.mean(num_grad[-3:])
    mean_grad_before_end = np.mean(num_grad[-50:-5])
    lamda = mean_grad_before_end / mean_grad_end
    if lamda < .25:
        print("\t\tλ ratio = ", lamda, " => Numerical solution diverges")
        return True
    print("\t\tλ ratio = ", lamda, " => Numerical solution does not diverge")
    return False


def save_metadata(date, case_prefix, lengths, real_physical_varied_ranges, omega0_vals, planetary_system, interesting_runs, further_notes=None):
    # lengths is essentially (10, 10, 10)
    file = open(f"data_storage/metadata_{case_prefix}_{date}.txt", "w")

    # Write a string to the file
    file.write(f"This is a run on the {date}\n")
    if further_notes is not None:
        file.write("This run corresponds to " + further_notes + "\n")
    file.write(f"A semi-major-axes cube of lengths {lengths} has been explored. \n")
    file.write(f"This corresponds to the physical semi-major-axes ranges of: \n\n")

    names = ["a_p", "a_m", "a_sm"]
    normalizations = [AU, LU, SLU]
    units = ["AU", "LU", "SLU"]
    for name, normalization, physical_range, unit in zip(names, normalizations, real_physical_varied_ranges, units):
        file.write(name + "~" + str(physical_range/normalization))
        file.write(f"\t in units of {unit}.\n")

    file.write("\n\n")
    file.write("The fixed initial values of Ω_s, Ω_p, Ω_m of this search are:\n")
    for el in omega0_vals:
        file.write(str(el) + " --> " + str(1/el/3600) + " hours" + "\n")

    file.write("\n\nThe material properties of each body that were set in this iteration were:\n")
    for body in planetary_system:
        output = (
            f"Name: {body.name}\n"
            f"Mass: {body.mass}\n"
            f"Density in kg/m^3: {body.rho}\n"
            f"2nd Tidal Love Number: {body.k}\n"
            f"Quality factor: {body.Q}\n\n"
        )
        file.write(output)

    file.write("\n\nFurthermore, following interesting events have been clocked:\n")

    for key, events in interesting_runs.items():
        event_count = len(events)
        file.write(f"{key} ({event_count} # events)\n")

        if event_count == 0:
            file.write("\t---\n")
        else:
            for el in events:
                file.write(f"\t{el}\n")

        file.write("\n")  # Add a blank line for better readability

    # Close the file
    file.close()


def construct_debug_plot(debug, idc, cc, plot_dt_systematics, rest, plot_derivatives):
    """

    If the current iteration `id` equals to the iteration `cc`, information and plots for `cc` will be
    printed / displayed.

    :param debug: bool,     Whether or not to construct the debug plot.
    :param idc: tuple,      The 'identifier counter' triple: (outer_counter, middle_counter,
                            inner_counter) for each iteration.
    :param cc: tuple,       The 'control counter' triple: Specific numeric values.
    :param plot_dt_systematics: bool, Plots the evolution of the time difference between analytic
                                           and numeric divergence.
   :param rest: np.array       Other necessary quantites for plotting
   :param plot_derivatives: bool, Plots the found derivative state vectors on a secondary axis (right).

    time_points, sol, sol_object, y_init, planetary_system, list_of_std_mus = rest

    :return:
    """
    print("tracker.get_sign_changes()", tracker.get_sign_changes())
    if idc == cc and debug:
        print("\t\tConstructing debug plot at self set iteration.")
        time_points, sol, sol_object, y_init, planetary_system, list_of_std_mus = rest
        compare_numerics = False
        if compare_numerics:
            for index, TIME in enumerate(time_points):
                NUM = sol[0, :][index]
                ANA = \
                    semi_major_axes_analytical_solution(TIME, y_init, planetary_system, list_of_std_mus)[0]
                print("At time ", TIME, " the numeric solution is ", NUM, " which deviates from ", ANA,
                      " (analytic) by ", 100 * np.abs(ANA - NUM) / ANA, " %.")

            ANA_final_sm = \
                semi_major_axes_analytical_solution(time_points[-1], y_init, planetary_system,
                                                    list_of_std_mus)[0]
            abs_tol = 1e-6
            rel_tol = 1e-6
            tot_err = abs_tol + rel_tol * sol[0, :][-1]  # abs_tol * rel_tol * a_sm_final
            print(
                "\t the ERROR right now is kept smaller than atol + rtol * y which in the last case is equal to: ",
                tot_err, " but this already ", tot_err * 100 / ANA_final_sm,
                " % of the final analytic solution.")
        plt_results(sol_object=sol_object, print_sol_object=False, planetary_system=planetary_system,
                    list_of_mus=list_of_std_mus, show=True, save=False, plot_derivatives=plot_derivatives)
    elif plot_dt_systematics and idc == (10, 10, 10):
        # Really, what you should plot here is not the total number of steps, but the mean relative
        # step-size after the transition point.
        times = turn_seconds_to_years(np.array(t_arr_final), "Millions")  # t_arr_final is a global variable
        print("Times: ", times)
        print("Vals: ", delta_t_arr_div)
        # Create the primary plot
        fig, ax1 = plt.subplots()

        # Plot delta_t on the primary y-axis
        ax1.plot(times, delta_t_arr_div, "b.")
        ax1.set_xlabel("Time in Millions of years")
        ax1.set_ylabel(r"$\Delta t$ in years", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title("Evolution of divergence time points difference")

        # Create a secondary y-axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Plot number of steps on the secondary y-axis
        ax2.plot(times, num_steps_arr, "r.")
        ax2.set_ylabel("Number of steps", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.show()
    else:
        pass


def rerun_with_approximation(final_time, y_init, args, events, t_report):
    """
    If iteration was likely to be stiff, this function triggers.
    We now know that the analytical solution does not apply.

    Therefore, retry with hyperbolic tangent approximation and check closeness to non-approximated
    prior solution while skipping analytic solution check.
    :param final_time:
    :param y_init:
    :param args:
    :param events:
    :param t_report: where to report the solution at
    :return:
    """
    planetary_system, mu_m_sm, mu_p_m, mu_s_p = args
    sol_object = solve_ivp(
        fun=submoon_system_derivative_approximant,
        t_span=(0, final_time),
        y0=y_init, method="Radau",
        args=(planetary_system, mu_m_sm, mu_p_m, mu_s_p),
        events=events,
        rtol=1e-6,
        t_eval=t_report
        # jac=jacobian
    )
    return sol_object


def check_quality_of_approximation(tanh_solution, np_sign_solution):
    """
    Checks whether the numeric solution of the canonical DFEs (`np_sign_solution`) matches that
    of the hyperbolic tangent approximation (`tanh_solution`) up the maximum time point at which `np_sign_solution`
    was reported before the stiffness termination event kicked in (or whichever solution has the least number of
    points).
    This assumes that both `tanh_solution` and `np_sign_solution` have the same support points, which is the case
    if `t_eval` is specified in `solve_ivp`.
    WARNING: This check is only performed for the semi-major-axes of the bodies.
    Even if the code does not throw an error, there might still be significant deviations in the other variables that
    are not reported, since the Ω_i's are not associated with any termination event.
    """
    a_sm_sign = np_sign_solution[0, :]
    a_m_sign = np_sign_solution[1, :]
    a_p_sign = np_sign_solution[2, :]

    a_sm_tanh = tanh_solution[0, :]
    a_m_tanh = tanh_solution[1, :]
    a_p_tanh = tanh_solution[2, :]

    if len(a_sm_sign) < len(a_sm_tanh):
        sign_solution_smaller = True
    else:
        sign_solution_smaller = False

    if sign_solution_smaller:
        final_time_index = len(a_sm_sign) - 1

        a_sm_tanh = a_sm_tanh[:final_time_index + 1]
        a_m_tanh = a_m_tanh[:final_time_index + 1]
        a_p_tanh = a_p_tanh[:final_time_index + 1]
    else:
        final_time_index = len(a_sm_tanh) - 1

        a_sm_sign = a_sm_sign[:final_time_index + 1]
        a_m_sign = a_m_sign[:final_time_index + 1]
        a_p_sign = a_p_sign[:final_time_index + 1]


    rel_dev = np.array([np.abs(t-s)/s for t, s in zip([a_sm_sign, a_m_sign, a_p_sign], [a_sm_tanh, a_m_tanh, a_p_tanh])])

    if np.any(rel_dev > .01):
        print("\t\tERROR: The hyperbolic tangent approximator lead to relative deviations > 1% in the solutions. Array:"
              f"\t\t\nmax. deviation in each case (a_sm, a_m, a_p): ", [np.max(ar) for ar in rel_dev], ".")
        raise ValueError("tanh approximation deviation.")


def interesting_sign_flip_occured():
    """

    Extracts the signs of the solved gradients of the run of the y-state-vector and checks whether the gradients or positive
    definite or not. If not, it indicates a body, e.g., having fallen in and then migrating outwards again.
    This function should only be activated if the run was not stiff. Returns the iteration_counter.

    """
    num_of_sign_changes = np.array(tracker.get_sign_changes())
    if np.any(num_of_sign_changes > 0):
        return True
    else:
        return False




def solve_ivp_iterator(n_pix_planet: int, n_pix_moon: int, n_pix_submoon: int, y_init: list, planetary_system: list,
                       list_of_std_mus: list, use_initial_values=False, upper_lim_planet=30, lower_lim_planet=None,
                       debug_plot=False, further_notes = None, analyze_iter=False, specific_counter=None, case_prefix="",
                       force_tanh_approx=False) -> list:
    # noinspection StructuralWrap
    """
    The main function executing the numerical integration of the submoon system.
    The state of the system is encdoded in six variables that are ordered like this throughout the code:

    [submoon.a, moon.a, planet.a, moon.omega, planet.omega, star.omega].

    Return statement elements:

    -   `results`:

            Contains strings "-1", "+1" and "NUM ER" for each iteration indicating success status of integration-

    -   `lifetimes`, np.array with elements:

            [0]: (i, j, k,)
            [1]: turn_seconds_to_years(np.max(time_points))
            [2]: y_init
            [3]: y_final
            [4]: termination_reason
            [5]: whole_solution_object
            [6]: tracker.get_sign_changes()

            In words:
            [0] `iteration identifier` (natural numbers) that can be turned into real coordinates using the returned
            `physically_varied_ranges`, [1] the lifetime in this iteration in years, [2] the initial and [3] final state
             vector (None one if iteration ended prematurely), [4] the termination reason and [5] the whole solution
             object (None if there was none), [6] the number of sign changes in the semi-major-axis of submoon,
             moon and planet (likely due to stiffness most of the time, possibly because they are actual sign changes).

            Here, `iteration identifier` is a coordinate triple that specifies the edges of the grid search, e.g.
            (5 5 1).

    -   The rest should be clear


    :param n_pix_planet:       int,     The outer, planet semi-major-axis iteration step length.
    :param n_pix_moon:         int,     The middle, moon semi-major-axis iteration step length.
    :param n_pix_submoon:      int,     The inner, submoon semi-major-axis iteration step length.
    :param y_init:             list,    A list of floats that describe the initial values of the submoon system.
                                        Order: [submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0]
    :param planetary_system:   list,    A list of `CelestialBody` objects that represents the submoon system.
                                        Order: [star, planet, moon, submoon]
    :param list_of_std_mus:    list,    A list of floats containing the standard gravitational parameters of the system.
                                        Order: [mu_m_sm, mu_p_m, mu_s_p]
    :param use_initial_values: bool,    See documentation of `bind_system_gravitationally`
    :param upper_lim_planet:   float,   The upper limit of the planet's semi-major-axis to vary to in astronomical
                                        units.
    :param lower_lim_planet:   float,   The lower limit of the planet's s.m.axis.
    implicitly in AU.
                                        If not specified, will use the star's Roche Limit.
    :param debug_plot:         bool,    If true, plots the celestial bodies' trajectories for each point of the grid.
    :param analyze_iter:       bool,    If true, does not vary the semi-major-axes but instead just does what the
                                        code would do in the i-j-k-th iteration if parameter `specific_counter` is
                                        (i,j,k).
    :param specific_counter: tuple,     Tuple of three numbers indicating for which itereation counter to execute
                                        the code (see `analyze_iter` variable). i, j, k should be the value of the
                                        outer, middle and inner counter respectively.
    :param force_tanh_approx: bool,     If true, after each iteration of the np.sign differential equations, the
                                        np.tanh differential equations are simulated too, independent of whether
                                        the np.sign DFEs solutions ended up stiff. Very useful for analysis
                                        of the difference between np.sign and np.tanh solutions.

    :return:                   tuple,   Containing information objects for all iterations:
                                        results, lifetimes, termination_reason_counter, physically_varied_ranges

    """

    start_time = time.time()
    results = []  # contains strings "-1", "+1" and "NUM ER" for each iteration,
    # indicating success status of integration
    lifetimes = []
    interesting_runs = {"tanh() model deviated early on from signum() model: ": [],
                        "Sign flip in any quantity without being stiff: ": [],
                        "Submoon fell under roche limit": [],
                        "Submoon exceeded a_crit": [],
                        "Moon fell under roche limit": [],
                        "Moon exceeded a_crit": [],
                        "Bad initial values input": [],
                        "None": [],
                        "Some roche limit was greater than a_crit": [],
                        "Some initial value was under the roche or over the a_crit limit": [],
                        "Iteration is likely to be stiff": [],
                        }
    termination_reason_counter = {"Submoon fell under roche limit": 0,
                                  "Submoon exceeded a_crit": 0,
                                  "Moon fell under roche limit": 0,
                                  "Moon exceeded a_crit": 0,
                                  "Bad initial values input": 0,
                                  "None": 0,
                                  "Some roche limit was greater than a_crit": 0,
                                  "Some initial value was under the roche or over the a_crit limit": 0,
                                  "No termination event occurred.": 0,
                                  "Iteration is likely to be stiff": 0,
                                  }

    solve_ivp_iterator_console_logger(planetary_system, mode=0)
    raise_warning("Please ensure the correct order of the initial state:\n "
                  "[submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0]\n")

    # Unpack the planetary system
    star, planet, moon, submoon = planetary_system

    # Start iteration of the planet's semi-major-axis from its roche limit to 30 AU (approximately Neptune's distance)
    lower_lim_out = planet.get_current_roche_limit()
    if lower_lim_planet is not None:
        lower_lim_out = lower_lim_planet * AU
    upper_lim_out = upper_lim_planet * AU

    if analyze_iter:
        if specific_counter is None:
            raise ValueError("Specific counter is required for analyze_iter=True")
        outer_spec_counter = specific_counter[0]
        middle_spec_counter = specific_counter[1]
        inner_spec_counter = specific_counter[2]
    # Note: During this loop, the semi-major-axes and spin-frequencies of all updates are updated dynamically.
    outer_counter = 0
    physical_outer_range = np.linspace(lower_lim_out, upper_lim_out, n_pix_planet)
    for i in physical_outer_range:
        outer_counter += 1
        if analyze_iter:
            if outer_counter != outer_spec_counter:
                continue

        # Inject the variable to actually vary (planet's semi-major-axis)
        # The values of y_init stay constant throughout all simulations, except the quantity that is varied.
        y_init[2] = i

        # Update the value of the variable to vary so correct event functions are fired
        planet.update_semi_major_axis_a(i)

        # Start iteration of the moon's semi-major-axis from its roche limit to its critical semi-major-axis
        lower_lim_middle = moon.get_current_roche_limit()
        upper_lim_middle = moon.get_current_critical_sm_axis()

        middle_counter = 0
        physical_middle_range = np.linspace(lower_lim_middle, upper_lim_middle, n_pix_moon)
        for j in physical_middle_range:
            middle_counter += 1
            if analyze_iter:
                if middle_counter != middle_spec_counter:
                    continue

                    # Inject the variable to actually vary (moon's semi-major-axis)
            # The values of y_init stay constant throughout all simulations, except the quantity that is varied.
            y_init[1] = j

            # Update the value of the variable to vary so correct event functions are fired
            moon.update_semi_major_axis_a(j)

            # Start the iteration of the submoon's semi-major-axis from its roche limit to its critical semi-major-axis.
            lower_lim_in = submoon.get_current_roche_limit()
            upper_lim_in = submoon.get_current_critical_sm_axis()

            inner_counter = 0
            physical_inner_range = np.linspace(lower_lim_in, upper_lim_in, n_pix_submoon)
            for k in physical_inner_range:
                inner_counter += 1
                identifier = (outer_counter, middle_counter, inner_counter)

                if analyze_iter:
                    if inner_counter != inner_spec_counter:
                        continue

                tracker.iter_counter = (outer_counter, middle_counter, inner_counter)  # For debugging

                # Inject the second variable to vary (submoon's semi-major-axis)
                y_init[0] = k
                submoon.update_semi_major_axis_a(k)

                # Log some more stuff.
                solve_ivp_iterator_console_logger(planetary_system, mode=1, current_y_init=y_init,
                                                  upper_lim_out=upper_lim_out, upper_lim_middle=upper_lim_middle,
                                                  upper_lim_in=upper_lim_in, outer_counter=outer_counter,
                                                  middle_counter=middle_counter, inner_counter=inner_counter)

                # Sanity-check that e.g., roche limit is not bigger than critical-semi-major axis etc. (bad input).
                try:
                    sanity_check_initial_values(a_sm_0=y_init[0], a_m_0=y_init[1], a_p_0=y_init[2],
                                                planetary_system=planetary_system)


                except InitialValuesOutsideOfLimits as er:
                    lifetimes.append([(outer_counter, middle_counter, inner_counter), 0, y_init, None,
                                      "Some initial value was under the roche or over the a_crit limit",
                                      None, tracker.get_sign_changes()])  # See docstring for explanation
                    premature_termination_logger(er)
                    termination_reason_counter["Some initial value was under the roche or over the a_crit limit"] += 1
                    results.append("-1")
                    continue  # Continue to the next j iteration

                except RocheLimitGreaterThanCriticalSemiMajor as er:
                    lifetimes.append([(outer_counter, middle_counter, inner_counter), 0, y_init, None,
                                      "Some roche limit was greater than a_crit",
                                      None, tracker.get_sign_changes()])
                    results.append("-1")
                    lifetimes.extend([[(outer_counter, middle_counter, sm_pix), 0, y_init, None,
                                       "Some roche limit was greater than a_crit", None, tracker.get_sign_changes()]
                                      for sm_pix in range(inner_counter + 1, n_pix_submoon + 1)])  # Skip all other
                    # iterations as well
                    results.extend(["-1"] * len(list(range(inner_counter + 1, n_pix_submoon + 1))))
                    premature_termination_logger(er)
                    termination_reason_counter["Some roche limit was greater than a_crit"] += 1
                    break  # Break all of the k iterations and continue to the next j iteration, which
                    # completely determines the value of the roche limit and critical semi-major-axis.
                    # In that case, the j-loop (moon-loop) cannot ever host a stable submoon (k-loop) and in that case
                    # we should also fill in the rest of the k-loop as unstable after breaking it.

                # Finalize the system
                planetary_system = bind_system_gravitationally(planetary_system=planetary_system,
                                                               use_initial_values=use_initial_values, verbose=False)

                # define events to track and values to update and a list describing the events
                list_of_all_events = [update_values, track_sm_m_axis_1, track_sm_m_axis_2,
                                      track_m_p_axis_1, track_m_p_axis_2, check_if_stiff]
                keys = ["Values were updated.", "Submoon fell under roche limit", "Submoon exceeded a_crit",
                        "Moon fell under roche limit", "Moon exceeded a_crit", "Iteration is likely to be stiff"]

                # evolve to 4.5 Bn. years
                final_time = turn_billion_years_into_seconds(4.5)
                celestial_time = np.linspace(0, final_time, int(1e5))

                # Unpack standard gravitational parameters
                mu_m_sm, mu_p_m, mu_s_p = list_of_std_mus

                # Solve the problem
                tracker.clear()
                analytical_check_possibly_applicable = True
                sol_object = solve_ivp(
                    fun=submoon_system_derivative,
                    t_span=(0, final_time),
                    y0=y_init, method="Radau",
                    args=(planetary_system, mu_m_sm, mu_p_m, mu_s_p),
                    events=list_of_all_events,
                    rtol=1e-4,
                    t_eval=celestial_time
                    # jac=jacobian
                )

                # Unpack solution
                time_points, sol, t_events, _, _, _, _, status, message, success = unpack_solve_ivp_object(sol_object)
                y_final = sol[:, -1]

                # If termination event occurred, find termination reason
                termination_reason = find_termination_reason(status=status, t_events=t_events, keys=keys)
                termination_reason_old = termination_reason

                # Manual override to force tanh iteration for each iteration (pickle objects called "Gamma")
                if force_tanh_approx:
                    termination_reason = "Iteration is likely to be stiff"

                if termination_reason == "Iteration is likely to be stiff":
                    analytical_check_possibly_applicable = False
                    print("\t\tIteration is likely to be stiff, rerunning with hyperbolic tangent approximation.")
                    # Rename old values
                    (time_points_old, y_final_old, sol_old, t_events_old, status_old, message_old, success_old,
                     sol_object_old) = \
                        (time_points, y_final, sol, t_events, status, message, success, sol_object)
                    arguments = (planetary_system, mu_m_sm, mu_p_m, mu_s_p)

                    # Clear tracker
                    tracker.clear()

                    # `check if stiff` event can be explicitly removed if desired for debugging purposes
                    # (going to :-1 in `list_of_all_events` in the line below)
                    sol_object = rerun_with_approximation(final_time, y_init, arguments, list_of_all_events,
                                                          celestial_time)

                    time_points, sol, t_events, _, _, _, _, status, message, success = unpack_solve_ivp_object(sol_object)
                    y_final = sol[:, -1]

                    old_state = (time_points_old, sol_old, submoon_system_derivative)
                    termination_reason = find_termination_reason(status=status, t_events=t_events, keys=keys)
                    if termination_reason == "Iteration is likely to be stiff":
                        # NOTE: Check whether stiffness detection was removed in the rerun.
                        print("\t\tRerun with hyperbolic approximation ended up stiff as well. Plotting disabled.")
                        # plt_results(sol_object, planetary_system, list_of_std_mus, old_state=old_state)

                    try:
                        check_quality_of_approximation(sol, sol_old)
                    except ValueError:

                        enable_plot = False
                        if enable_plot:
                            mode = "enabled"
                        else:
                            mode = "disabled"

                        print("\t\tRerun with hyperbolic approximation ended deviating too much from stiff solution. "
                              f"Plotting {mode}.")
                        # Plot if necessary comparison of old (exact) and new solution (approximation)

                        if enable_plot:
                            if not analyze_iter:
                                # Else will get plotted twice
                                plt_results(sol_object, planetary_system, list_of_std_mus, old_state=old_state)

                        if analyze_iter:
                            print("`analyze_iter` has been set and tanh approximation kicked in. Plotting np.sign (purple)\n "
                                  "and np.tanh solutions (multicolor), before reverting back to old solution object.")
                            plt_results(sol_object, planetary_system, list_of_std_mus, old_state=old_state)

                        # Approximation not good enough => revert back to stiff solution object
                        time_points, sol, t_events, status, message, success, y_final, sol_object = \
                            (time_points_old, sol_old, t_events_old, status_old, message_old, success_old, y_final_old, sol_object_old)
                        termination_reason = find_termination_reason(status=status, t_events=t_events, keys=keys)
                        print( "\t\t! Reverting back to old solution object (without tanh approximation). !")

                        interesting_runs["tanh() model deviated early on from signum() model: "].extend([identifier])

                    if analyze_iter:
                        # Even if solution passed quality inspection, plot tanh - sgn evolutions
                        print("\tPlotting sgn-tanh general comparison.")
                        plt_results(sol_object, planetary_system, list_of_std_mus, old_state=old_state)




                control_counter = (None, None, None)
                if analyze_iter and termination_reason != "Iteration is likely to be stiff":
                    # If termination reason was stiff, plot has already been constructed
                    control_counter = specific_counter
                    construct_debug_plot(debug=debug_plot, idc=identifier, cc=control_counter, plot_dt_systematics=True,
                                         rest=(time_points, sol, sol_object, y_init, planetary_system, list_of_std_mus),
                                         plot_derivatives=False)


                if interesting_sign_flip_occured():
                    interesting_runs["Sign flip in any quantity without being stiff: "].extend([identifier])

                # Numerical solution sanity-check I
                sanity_check_tot_ang_momentum_evolution(mu_m_sm=mu_m_sm, mu_p_m=mu_p_m, mu_s_p=mu_s_p,
                                                        planetary_system=planetary_system,
                                                        y_state_vector_evolution=sol.T)

                # Numerical solution sanity-check II
                if analytical_check_possibly_applicable:
                    sanity_check_with_analytical_sol_if_possible(mu_m_sm=mu_m_sm, mu_p_m=mu_p_m, mu_s_p=mu_s_p,
                                                                 planetary_system=planetary_system,
                                                                 y_state_vector_evolution=sol.T, time_points=time_points)

                # Document results
                document_result(status=status, termination_reason=termination_reason, time_points=time_points,
                                results=results, i=outer_counter, j=middle_counter, k=inner_counter, y_init=y_init,
                                y_final=y_final, termination_reason_counter=termination_reason_counter,
                                lifetimes=lifetimes, whole_solution_object=sol_object,
                                interesting_events_dict=interesting_runs)

    consistency_check_volume(expected_volume=n_pix_planet * n_pix_moon * n_pix_submoon, actual_volume=len(lifetimes))



    # The values of the physical axes that have been grid searched
    physically_varied_ranges = [physical_outer_range, physical_middle_range, physical_inner_range]

    # Get computation time
    end_time = time.time()
    computation_time = end_time - start_time

    # Return object
    data = (results, termination_reason_counter, lifetimes, physically_varied_ranges, computation_time)

    if not analyze_iter:
        now = datetime.datetime.now()
        pickle_me_this(
            f'data_storage/{case_prefix}_integration_{now}', data)

        save_metadata(now, case_prefix,(n_pix_planet, n_pix_moon, n_pix_submoon), physically_varied_ranges,
                      y_init[-3:], planetary_system, interesting_runs, further_notes)

    return data

