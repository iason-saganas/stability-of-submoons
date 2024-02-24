import numpy as np
from scipy.constants import G
from warnings import warn as raise_warning
import pandas as pd
from typing import Union

__all__ = ['check_if_direct_orbits', 'keplers_law_n_from_a', 'keplers_law_a_from_n', 'get_standard_grav_parameter',
           'get_hill_radius_relevant_to_body', 'get_critical_semi_major_axis', 'get_roche_limit',
           'analytical_lifetime_one_tide', 'dont', 'get_solar_system_planets_data','create_toy_satellite_and_planet',
           'CelestialBody']


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


def analytical_lifetime_one_tide(a_0: float, a_i: float, hosted_body: 'CelestialBody') -> float:
    """
    Calculates the time it takes for the semi-major-axis to reach `a_i` ,starting from `a_0` using the inputted set of
    parameters describing the system. This represents the analytical formula for the lifetime T in a one-tide-system,
    given by Murray and Dermott equation (4.213).

    Let j represent the satellite and i its hosting body. Then,

    T = 2/13 * a_0^(13/2) * ( 1-(a_i/a_0) ^ (13/2) ) * ( 3k_{2i} / Q_i * R_i^5 * m_j * (G/m_i)^(1/2) )^(-1)

    :parameter a_0:         float,              The initial semi-major-axis of the satellite j.
    :parameter a_i:         float,              The semi-major-axis value of j to evolve to.
    :parameter hosted_body: CelestialBody,      The satellite j to evolve.
    :return: T:             float,              The analytically calculated time it took for the evolution.
    """
    j = hosted_body
    i = j.hosting_body
    left_hand_side = 2/13 * a_0**(13/2)*(1-(a_i/a_0)**(13/2))
    right_hand_side = 3 * i.k / i.Q * (G/i.m)**(1/2) * i.R**5 * j.m
    T = left_hand_side / right_hand_side
    return T

def dont():
    """
    Do nothing function.
    """
    pass


def get_solar_system_planets_data(planet_name: str = '', physical_property: str = '', print_return: bool = False) \
        -> Union[tuple, float]:
    """
    Reads the file contents 'constants/planets_solar_system.txt'.
    If `planet_name` is provided, data for that planet will be returned and can then be accessed via tuple unpacking,
    mass, semi_major_axis, etc... = get_solar_system_planets_data(signature).
    If `planet_name` and `physical_property` is provided, a float will be returned that represent the queried
    physical property.

    :param planet_name: str,        One of: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus or Neptune.
    :param physical_property: str,  One of: Mass, Semi-major-axis, Diameter, Orbital-Period, Orbital-eccentricity,
                                            Density, Rotation-Period.
    :param print_return: bool,      Print what is returned.
    """
    df = pd.read_csv('constants/planets_solar_system.txt')
    whole_row_mode = (planet_name != '' and physical_property == '')
    specific_value_mode = (planet_name != '' and physical_property != '')
    if whole_row_mode:
        row = df[df['Planet'] == planet_name]
        row_nice_representation = row.iloc[0]
        res = tuple(row_nice_representation)
        print(row_nice_representation) if print_return else dont()
        return res
    elif specific_value_mode:
        row = df[df['Planet'] == planet_name]
        row_nice_representation = row.iloc[0]
        all_column_names = df.columns.tolist()
        user_short_hand = physical_property
        selected_index = 0
        for index, column_name in enumerate(all_column_names):
            if user_short_hand in column_name:
                selected_index = index
                break
        res = row_nice_representation[all_column_names[selected_index]]
        print(res) if print_return else dont()
        return res
    else:
        print(df)


print(get_solar_system_planets_data(planet_name='Earth',print_return=True))


def create_toy_satellite_and_planet():
    (name_earth, m_earth, a_earth, diameter_earth, orbital_period_earth, eccentricity_earth, density_earth,
     rotation_period_earth) = get_solar_system_planets_data('Earth')
    spin_frequency_earth = 1/(3600 * rotation_period_earth)

    toy_planet = CelestialBody(mass=m_earth,density=density_earth, semi_major_axis=None, spin_frequency=spin_frequency_earth,
                               love_number=)
    pass


class CelestialBody:
    """
    Base class representing a celestial body with its various properties set as class attributes.

    Because a parameter 'hosting_body' is needed, when defining all celestial bodies, the star is the very
    first body that needs to be instantiated such that it can be passed as a parameter to the planet, which in turn
    can be passed to the moon's definition etc.

    On instantiation, the parameters `semi_major_axis` and `spin_frequency` should represent the initial values.

    Attributes:
    ----------------

    :parameter mass:                float,          Body's mass.
    :parameter density:             float,          The body's mean density.
    :parameter semi_major_axis:     Union[float, None],   Value for the semi-major-axis. On instantiation, this should be the
                                                    semi-major-axis initial value `a_0` and may then be updated through
                                                    the method 'update_semi_major_axis'.
    :parameter spin_frequency:      float,          Value for the spin-frequency. On instantiation, this should be the
                                                    spin-frequency initial value `omega_0` and may then be updated
                                                    through the method 'update_spin_frequency'.
    :parameter love_number:         float,          The second tidal love number associated with the body's rheology.
    :parameter quality_factor:      float,          The quality factor associated with the body's rheology.
    :parameter descriptive_index:   str,            A descriptive index shorthand for the body, e.g. "sm" for submoon.
    :parameter name:                str,            The name for the body, e.g. "submoon".
    :parameter hierarchy_number:    int,            The hierarchy number corresponding to the body's position in the
                                                    nested body system. 1 for star, 2 for planet, 3 for moon,
                                                    4 for submoon.
    :parameter hosting_body:        Union[float, None],  The body that `self` orbits, i.e. the hosting body.

    Methods:
    ----------------
    update_semi_major_axis_a:       Updates the semi-major-axis based on a value.
    update_spin_frequency_omega:    Updates the spin-frequency based on a value.
                                    semi-major-axis. After initialization, `self.n` can be used instead.
                                    `self` and its hosts as specified by `self.hosting_body`


    Properties
    (Can be accessed via dot notation of a class instance like attributes but are defined via a distinct class method
    instead of inside `__init__()`) :
    ----------------
    n:                              The current orbit-frequency calculated from the current semi-major-axis using
                                    Kepler's Third Law.
    mu:                             The standard gravitational parameter between `self` and the body that `self` orbits,
                                    specified by `self.hosting_body`.

    """

    def __init__(self, mass: float, density: float, semi_major_axis: Union[float, None], spin_frequency: float, love_number: float,
                 quality_factor: float, descriptive_index: str, name: str, hierarchy_number: int,
                 hosting_body: Union['CelestialBody', None]):
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
