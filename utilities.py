import numpy as np
from scipy.constants import G
from warnings import warn as raise_warning
import pandas as pd
from typing import Union, List
from scipy.integrate import solve_ivp
from style_components.matplotlib_style import *
import pickle
from style_components.voxel_plotter import plot_3d_voxels

AU = 1.496e11

__all__ = ['check_if_direct_orbits', 'keplers_law_n_from_a', 'keplers_law_a_from_n', 'keplers_law_n_from_a_simple',
           'get_standard_grav_parameter', 'get_hill_radius_relevant_to_body', 'get_critical_semi_major_axis',
           'get_roche_limit', 'analytical_lifetime_one_tide', 'dont', 'get_solar_system_bodies_data',
           'CelestialBody', 'turn_seconds_to_years', 'get_a_derivative_factors_experimental', 'get_a_factors',
           'get_omega_derivative_factors_experimental', 'get_omega_factors', 'unpack_solve_ivp_object',
           'turn_billion_years_into_seconds', 'bind_system_gravitationally', 'custom_experimental_plot',
           'submoon_system_derivative', 'update_values', 'track_submoon_sm_axis_1', 'track_submoon_sm_axis_2',
           'track_moon_sm_axis_1', 'track_moon_sm_axis_2', 'reset_to_default', 'solve_ivp_iterator', 'showcase_results',
           'pickle_me_this', 'unpickle_me_this']


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


def get_hill_radius_relevant_to_body(hosted_body: 'CelestialBody') -> float:
    """
    Gets the hill radius that is relevant to `hosted_body`, i.e. the gravitational sphere of influence exerted by the
    body that `hosted_body` orbits.

    Let i, j, k represent hierarchy numbers, with i < j < k, i.e. `i` is the most 'un-nested' body and `k` is very
    nested. `k` orbits `j` orbits `i`.

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
    m_ratio = .1
    a_ratio = .3
    for index, hn in enumerate(hn_list):
        if hn == 1:
            # No hosting body to check mass or semi-major-axis against if it's on the first hierarchy level
            pass
        else:
            hosted_body = planetary_system[index]
            hosting_body = hosted_body.hosting_body

            mass_ratio = hosted_body.mass / hosting_body.mass
            if mass_ratio > m_ratio:
                raise ValueError(f"\n Exception inside 'bind_system_gravitationally' function: \n"
                                 f"Mass ratio between bodies {hosting_body.name} and {hosted_body.name} is too big.\n "
                                 f"Detected mass ratio: {mass_ratio}. Threshold: {m_ratio} "
                                 f"(assumption that needs to be checked!).\n Absolute mass values: {hosting_body.mass} "
                                 f"and {hosted_body.mass} respectively. \n")
            elif verbose:
                print(f"Mass ratio sanity check between bodies {hosting_body.name} and {hosted_body.name} passed.\n "
                      f"            Detected mass ratio: {mass_ratio}. Threshold: {m_ratio}.\n\n")

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


def custom_experimental_plot(time_points, solution, derivative, planetary_system, list_of_mus):
    """
    :ToDo Beautify plots and write better documentation and make plotting code more concise
    :param time_points: The time points at which solutions were found
    :param solution: numpy array, the y solution coming out of 'solve_ivp'
    :param derivative: The derivative (callable)
    :return:
    """

    fig, axs = plt.subplots(3, 1, figsize=(15, 8))

    for y_sol, color, label, axis in zip(solution[:3], [red, green, blue],
                                         [r"$a_{\mathrm{sm}}$", r"$a_{\mathrm{m}}$", r"$a_{\mathrm{p}}$"], axs):
        axis.plot(time_points, y_sol, marker=".", color=color,
                  label=label + f' num of points: {len(y_sol)}')
        axis.plot(time_points, y_sol, linestyle="-", color=color)
        axis.legend()

    axs[2].set_xlabel(r"$t$ in $\mathrm{s}$")
    axs[2].set_ylabel("$a(t)$ in meters")
    axs[0].set_title('Evolution of Semi-Major-Axes of Bodies $a(t)$',
                     fontsize=16)  # Adjust fontsize and y position as needed
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(15, 8))

    for y_sol, color, label, axis in zip(solution[3:6], [red, green, blue],
                                         [r"$\Omega(t)$ moon", r"$\Omega(t)$ planet", r"$\Omega(t)$ star"], axs):
        axis.plot(time_points, y_sol * 1000, marker=".", color=color, label=label + f' num of points: {len(y_sol)}')
        axis.plot(time_points, y_sol * 1000, linestyle="-", color=color)
        axis.legend()

    axs[2].set_xlabel(r"$t$ in $\mathrm{s}$")
    axs[2].set_ylabel(r"$\Omega(t)$ in $\mathrm{mHz}$")
    axs[0].set_title(r'Evolution of Spin-Frequencies of Bodies $\Omega (t)$', fontsize=16,
                     y=1.05)  # Adjust fontsize and y position as needed
    plt.tight_layout()
    plt.show()

    dy_dt = np.array(derivative(time_points, solution, planetary_system, *list_of_mus))

    # Plot the derivative values against time
    fig, axs = plt.subplots(3, 1, figsize=(15, 8))

    for y_sol, color, label, axis in zip(dy_dt[:3], [red, green, blue], [r"$\frac{\mathrm{d}a}{\mathrm{d}t}$ submoon",
                                                                         r"$\frac{\mathrm{d}a}{\mathrm{d}t}$ moon",
                                                                         r"$\frac{\mathrm{d}a}{\mathrm{d}t}$ planet"],
                                         axs):
        axis.plot(time_points, 100 * y_sol * (365 * 24 * 3600), marker=".", color=color,
                  label=label + f' num of points: {len(y_sol)}')
        axis.plot(time_points, 100 * y_sol * (365 * 24 * 3600), linestyle="-", color=color)
        axis.legend()

    axs[2].set_xlabel(r"$t$ in $\mathrm{s}$")
    axs[2].set_ylabel(r"$\dot{a}(t)$ in $\mathrm{\frac{cm}{yr}}$")
    axs[0].set_title(r'Evolution of Rate of Change of Semi-Major-Axes of Bodies $\frac{\mathrm{d}a}{\mathrm{d}t} (t)$',
                     fontsize=16,
                     y=1.05)  # Adjust fontsize and y position as needed
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(15, 8))

    for y_sol, color, label, axis in zip(dy_dt[3:6], [red, green, blue],
                                         [r"$\frac{\mathrm{d}\Omega}{\mathrm{d}t}$ moon",
                                          r"$\frac{\mathrm{d}\Omega}{\mathrm{d}t}$ planet",
                                          r"$\frac{\mathrm{d}\Omega}{\mathrm{d}t}$ star"],
                                         axs):
        axis.plot(time_points, y_sol * 1000, marker=".", color=color, label=label + f' num of points: {len(y_sol)}')
        axis.plot(time_points, y_sol * 1000, linestyle="-", color=color)
        axis.legend()

    axs[2].set_xlabel(r"$t$ in $\mathrm{s}$")
    axs[2].set_ylabel(r"$\dot{\Omega}(t)$ in $\mathrm{mHz}$")
    axs[0].set_title(r'Evolution of Rate of Change of Spin-Frequencies of '
                     r'Bodies $\frac{\mathrm{d}\Omega}{\mathrm{d}t}(t)$', fontsize=16,
                     y=1.05)  # Adjust fontsize and y position as needed
    plt.tight_layout()
    plt.show()


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
    a_m_sm_dot = get_a_factors(submoon) * np.sign(omega_m - n_m_sm) * a_m_sm ** (-11 / 2)
    a_p_m_dot = get_a_factors(moon) * np.sign(omega_p - n_p_m) * a_p_m ** (-11 / 2)
    a_s_p_dot = get_a_factors(planet) * np.sign(omega_s - n_s_p) * a_s_p ** (-11 / 2)

    # Define the spin-frequency derivatives
    omega_m_dot = (-get_omega_factors(moon) * (np.sign(omega_m - n_p_m) * planet.mass ** 2 * a_p_m ** (-6)
                                               + np.sign(omega_m - n_m_sm) * submoon.mass ** 2 * a_m_sm ** (
                                                   -6)))
    omega_p_dot = (- get_omega_factors(planet) * (np.sign(omega_p - n_s_p) * star.mass ** 2 * a_s_p ** (-6)
                                                  + np.sign(omega_p - n_p_m) * moon.mass ** 2 * a_p_m ** (-6)))
    omega_s_dot = - get_omega_factors(star) * np.sign(omega_s - n_s_p) * planet.mass ** 2 * a_s_p ** (-6)

    # Define and return the derivative vector
    dy_dt = [a_m_sm_dot, a_p_m_dot, a_s_p_dot, omega_m_dot, omega_p_dot, omega_s_dot]
    return dy_dt


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

    # Unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system

    star.update_spin_frequency_omega(omega_s)
    planet.update_spin_frequency_omega(omega_p)
    moon.update_spin_frequency_omega(omega_p)

    planet.update_semi_major_axis_a(a_s_p)
    moon.update_semi_major_axis_a(a_p_m)
    submoon.update_semi_major_axis_a(a_m_sm)

    # Use an infinity value, so to not actually activate the event
    return omega_s - np.inf


def track_submoon_sm_axis_1(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    rl = submoon.get_current_roche_limit()
    # if a_m_sm - rl < 0:
    #    print("\t\tThe submoon's semi-major-axis fell under the moon's roche limit.")
    #    print("val of a_sm: ", a_m_sm, " roche limit: ", rl)
    return a_m_sm - rl


def track_submoon_sm_axis_2(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    crit_a = submoon.get_current_critical_sm_axis()
    # if a_m_sm - crit_a > 0:
    #    print("\t\tThe submoon's semi-major-axis surpassed the critial semi-major-axis. Val of current axis:", a_m_sm,
    #          " and val of current crit a ", crit_a)
    return a_m_sm - crit_a


def track_moon_sm_axis_1(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    rl = moon.get_current_roche_limit()
    # if a_p_m - rl < 0:
    #     print("\t\tThe moons's semi-major-axis fell under the planets's roche limit.")
    return a_p_m - rl


def track_moon_sm_axis_2(t, y, planetary_system: List['CelestialBody'], mu_m_sm, mu_p_m, mu_s_p):
    # Unpack all values
    a_m_sm, a_p_m, a_s_p, omega_m, omega_p, omega_s = y
    # List unpack the celestial bodies of the system to access their pre-defined properties
    star, planet, moon, submoon = planetary_system
    crit_a = moon.get_current_critical_sm_axis()
    # if a_p_m - crit_a > 0:
    #     print("\t\tThe moons's semi-major-axis surpassed the critical semi-major-axis.")
    return a_p_m - crit_a


update_values.terminal = False
track_submoon_sm_axis_1.terminal = True
track_submoon_sm_axis_2.terminal = True
track_moon_sm_axis_1.terminal = True
track_moon_sm_axis_2.terminal = True


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
    Sanity checks if the initial values relevant for termination of the simulation (semi-major-axis of moon and submoon
    and planet) don't from the get-go surpass or fall under their respective a_crit or roche-limit.
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

    print("\n------------------------------------------\n")
    return stability_status


def plt_results(sol_object, planetary_system, list_of_mus, print_sol_object=False):
    # Unpack variables outputted by solution object.
    (time_points, solution, t_events, y_events, num_of_eval, num_of_eval_jac, num_of_lu_decompositions, status,
     message, success) = unpack_solve_ivp_object(sol_object)

    if print_sol_object:
        print("sol_object : ", sol_object)
    custom_experimental_plot(time_points, solution, submoon_system_derivative, planetary_system, list_of_mus)


def find_termination_reason(status, t_events, keys):
    if status == 1:
        term_index = [i for i, array in enumerate(t_events) if bool(len(array))]
        if len(term_index) > 1:
            raise ValueError("More than one termination event")
        return keys[term_index[0]]
    else:
        return "No termination event occurred."


def document_result(status, termination_reason, time_points, results, i, j, k,
                    y_init, y_final, termination_reason_counter, lifetimes, whole_solution_object):
    stability_status = log_results(status, termination_reason, time_points)
    results.append(stability_status)
    lifetimes.append([(i, j, k,), turn_seconds_to_years(np.max(time_points)), y_init, y_final,
                      termination_reason, whole_solution_object])
    termination_reason_counter[str(termination_reason)] += 1


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


def showcase_results(result, n_pix_planet=None, n_pix_moon=None, n_pix_submoon=None):
    from collections import defaultdict

    hits, termination_reason_counter, lifetimes = result

    hit_counts = defaultdict(int)
    for hit in hits:
        hit_counts[hit] += 1

    print("\n\n--------RESULTS--------\n\n")
    print("\n------ Stability states (-1: Termination, NUM ER: Numerical error, +1: Longevity):")
    for key, value in hit_counts.items():
        print(f'{key}: {value}')
    print("\n------")

    print("\n------ Histogram of termination reasons:")
    for key, value in termination_reason_counter.items():
        print(f'{key}: {value}')
    print("\n------")

    print("\n------ Longest lifetime object (Second element is lifetime in years):")
    max_subarray = max(lifetimes, key=lambda x: x[1])
    index_of_longest_lifetime_array = lifetimes.index(max_subarray)
    print(lifetimes[index_of_longest_lifetime_array])

    if n_pix_planet is not None:
        if any((n_pix_moon, n_pix_submoon)) is None:
            raise ValueError("n_pix_moon and n_pix_submoon have to be provided")

        normalization = 4.5e9  # years
        coordinates = np.array([item[0] for item in lifetimes])
        values = np.array([item[1] for item in lifetimes])/normalization

        # Concatenate the arrays along the second axis
        merge = np.hstack((coordinates, np.expand_dims(values, axis=1)))

        # Create an empty Xdim x Ydim x Zdim x1 array
        full_4D_data = np.zeros((n_pix_submoon, n_pix_moon, n_pix_submoon, 1))
        # Iterate over each row and assign color channel value
        for row in merge:
            x, y, z, color = row
            full_4D_data[int(x) - 1, int(y) - 1, int(z) - 1, 0] = color

        plot_3d_voxels(data=full_4D_data, multi_view=False, sampling_ratio=1, skew_factor=1, colormap='RdYlBu_r')


def solve_ivp_iterator(n_pix_planet: int, n_pix_moon: int, n_pix_submoon: int, y_init: list, planetary_system: list,
                       list_of_std_mus: list, use_initial_values=False, upper_lim_planet=30) -> list:

    # noinspection StructuralWrap
    """
    The main function executing the numerical integration of the submoon system.
    The state of the system is encdoded in six variables that are ordered like this throughout the code:

    [submoon.a, moon.a, planet.a, moon.omega, planet.omega, star.omega],

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

    :return:                   tuple,   Containing information objects for all iterations:
                                        results, lifetimes, termination_reason_counter

    """
    results = []  # contains strings "-1", "+1" and "NUM ER" for each iteration,
    # indicating success status of integration
    lifetimes = []  # contains elements like this:
    # ((iteration identifier),lifetime in years, [initial state vector],
    # [last state vector], termination reason, whole solution object from solve_ivp)
    termination_reason_counter = {"Submoon fell under roche limit": 0,
                                  "Submoon exceeded a_crit": 0,
                                  "Moon fell under roche limit": 0,
                                  "Moon exceeded a_crit": 0,
                                  "Bad initial values input": 0,
                                  "None": 0,
                                  "Some roche limit was greater than a_crit": 0,
                                  "Some initial value was under the roche or over the a_crit limit": 0,
                                  "No termination event occurred.": 0}

    solve_ivp_iterator_console_logger(planetary_system, mode=0)
    raise_warning("Please ensure the correct order of the initial state:\n "
                  "[submoon.a0, moon.a0, planet.a0, moon.omega0, planet.omega0, star.omega0]\n")

    # Unpack the planetary system
    star, planet, moon, submoon = planetary_system

    # Start iteration of the planet's semi-major-axis from its roche limit to 30 AU (approximately Neptune's distance)
    lower_lim_out = planet.get_current_roche_limit()
    upper_lim_out = upper_lim_planet*AU

    # Note: During this loop, the semi-major-axes and spin-frequencies of all updates are updated dynamically.
    # Do not reference 'submoon.a' with the expectation to access the value that was assigned by
    # 'create_submoon_system'.
    # Exception: After each simulation, i.e., the end of 'solve_ivp', all values are reset to the values assigned by
    # 'create_submoon_system'.
    outer_counter = 0
    for i in np.linspace(lower_lim_out, upper_lim_out, n_pix_planet):
        outer_counter += 1

        # Inject the variable to actually vary (moon's semi-major-axis)
        # The values of y_init stay constant throughout all simulations, except the quantity that is varied.
        y_init[2] = i

        # Update the value of the variable to vary so correct event functions are fired
        planet.update_semi_major_axis_a(i)

        # Start iteration of the moon's semi-major-axis from its roche limit to its critical semi-major-axis
        lower_lim_middle = moon.get_current_roche_limit()
        upper_lim_middle = moon.get_current_critical_sm_axis()

        middle_counter = 0
        for j in np.linspace(lower_lim_middle, upper_lim_middle, n_pix_moon):
            middle_counter += 1

            # Inject the variable to actually vary (moon's semi-major-axis)
            # The values of y_init stay constant throughout all simulations, except the quantity that is varied.
            y_init[1] = j

            # Update the value of the variable to vary so correct event functions are fired
            moon.update_semi_major_axis_a(j)

            # Start the iteration of the submoon's semi-major-axis from its roche limit to its critical semi-major-axis.
            lower_lim_in = submoon.get_current_roche_limit()
            upper_lim_in = submoon.get_current_critical_sm_axis()

            inner_counter = 0
            for k in np.linspace(lower_lim_in, upper_lim_in, n_pix_submoon):
                inner_counter += 1

                # Inject the second variable to vary (submoon's semi-major-axis)
                y_init[0] = k
                submoon.update_semi_major_axis_a(k)

                # Log some more stuff.
                solve_ivp_iterator_console_logger(planetary_system, mode=1, current_y_init=y_init,
                                                  upper_lim_out=upper_lim_out, upper_lim_middle=upper_lim_middle,
                                                  upper_lim_in=upper_lim_in, outer_counter=outer_counter,
                                                  middle_counter=middle_counter,  inner_counter=inner_counter)

                # Sanity-check that e.g., roche limit is not bigger than critical-semi-major axis etc. (bad input).
                try:
                    sanity_check_initial_values(a_sm_0=y_init[0], a_m_0=y_init[1], a_p_0=y_init[2],
                                                planetary_system=planetary_system)
                except InitialValuesOutsideOfLimits as er:
                    lifetimes.append([(outer_counter, middle_counter, inner_counter), 0])  # coordinates, 0 years stable
                    premature_termination_logger(er)
                    termination_reason_counter["Some initial value was under the roche or over the a_crit limit"] += 1
                    continue  # Continue to the next j iteration
                except RocheLimitGreaterThanCriticalSemiMajor as er:
                    lifetimes.append([(outer_counter, middle_counter, inner_counter), 0])  # coordinates, 0 years stable
                    premature_termination_logger(er)
                    termination_reason_counter["Some roche limit was greater than a_crit"] += 1
                    break  # Break all of the k iterations and continue to the next j iteration, which
                    # completely determines the value of the roche limit and critical semi-major-axis.
                    # In that case, the j-loop (moon-loop) cannot ever host a stable submoon (k-loop)

                # Finalize the system
                planetary_system = bind_system_gravitationally(planetary_system=planetary_system,
                                                               use_initial_values=use_initial_values, verbose=False)

                # define events to track and values to update and a list describing the events
                list_of_all_events = [update_values, track_submoon_sm_axis_1, track_submoon_sm_axis_2,
                                      track_moon_sm_axis_1, track_moon_sm_axis_2]
                keys = ["Values were updated.", "Submoon fell under roche limit", "Submoon exceeded a_crit",
                        "Moon fell under roche limit", "Moon exceeded a_crit"]

                # evolve to 4.5 Bn. years
                final_time = turn_billion_years_into_seconds(4.5)

                # Unpack standard gravitational parameters
                mu_m_sm, mu_p_m, mu_s_p = list_of_std_mus

                # Solve the problem
                # noinspection PyTupleAssignmentBalance
                sol_object = solve_ivp(
                    fun=submoon_system_derivative,
                    t_span=(0, final_time),
                    y0=y_init, method="RK23",
                    args=(planetary_system, mu_m_sm, mu_p_m, mu_s_p),
                    events=list_of_all_events,
                )

                # Unpack solution
                time_points, sol, t_events, _, _, _, _, status, message, success = unpack_solve_ivp_object(sol_object)
                y_final = sol[:, -1]

                # If termination event occurred, find termination reason
                termination_reason = find_termination_reason(status=status, t_events=t_events, keys=keys)

                # Document results
                document_result(status=status, termination_reason=termination_reason, time_points=time_points,
                                results=results, i=outer_counter,j=middle_counter, k=inner_counter, y_init=y_init,
                                y_final=y_final, termination_reason_counter=termination_reason_counter,
                                lifetimes=lifetimes, whole_solution_object=sol_object)

                plot = False
                if plot:
                    plt_results(sol_object=sol_object, print_sol_object=False, planetary_system=planetary_system,
                                list_of_mus=list_of_std_mus)

    pickle_me_this(f'data_storage/Integration. Num of Pixels (pl, moon, sm) = {n_pix_planet, n_pix_moon, n_pix_submoon}; '
                   f'omega_init (moon, pl, star) = {y_init[-3:]}', (results, termination_reason_counter, lifetimes))
    return results, termination_reason_counter, lifetimes
