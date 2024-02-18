from utilities import keplers_law_n_from_a, check_if_direct_orbits, get_standard_grav_parameter
import numpy as np


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
    :parameter density:                float,       The body's mean density.
    :parameter semi_major_axis:     float,          Value for the semi-major-axis. On instantiation, this should be the
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
    :parameter hosting_body:        CelestialBody,  The body that `self` orbits, i.e. the hosting body.

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

    def __init__(self, mass: float, density: float, semi_major_axis: float, spin_frequency: float, love_number: float,
                 quality_factor: float, descriptive_index: str, name: str, hierarchy_number: int,
                 hosting_body: 'CelestialBody'):
        self.mass = mass
        self.rho = density
        self.a = semi_major_axis
        self.omega = spin_frequency
        self.k = love_number
        self.Q = quality_factor
        self.descriptive_index = descriptive_index
        self.name = name
        self.hn = hierarchy_number

        if hierarchy_number == 1:
            # Star has no hosting body.
            self.hosting_body = None
        else:
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
