from utilities import *
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['create_toy_satellite_and_planet', 'create_earth_submoon_system', 'create_warm_jupiter_submoon_system']


def create_toy_satellite_and_planet():
    """
    :ToDo Write documentation.
    :return:
    """
    # get some parameters from solar system bodies. The '_d' stands for '_data'.
    earth_d = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                           name_of_celestial_body='Earth')

    luna_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                          name_of_celestial_body='Moon')

    # define spin frequencies
    spin_frequency_earth = 1 / (3600 * earth_d.T_rotation_hours)
    spin_frequency_luna = 1 / (3600 * luna_d.T_rotation_hours)

    # define first toy planet
    toy_planet = CelestialBody(mass=earth_d.m, density=earth_d.rho, semi_major_axis=None,
                               spin_frequency=spin_frequency_earth, love_number=earth_d.k,
                               quality_factor=earth_d.Q, descriptive_index="p", name="toy planet",
                               hierarchy_number=1, hosting_body=None)

    # then toy satellite
    toy_satellite = CelestialBody(mass=luna_d.m, density=luna_d.rho, semi_major_axis=luna_d.a,
                                  spin_frequency=spin_frequency_luna, love_number=luna_d.j,
                                  quality_factor=luna_d.Q, descriptive_index="s", name="toy satellite",
                                  hierarchy_number=2, hosting_body=toy_planet)

    return toy_satellite, toy_planet


def create_earth_submoon_system(P_rot_star_DAYS, P_rot_planet_HOURS, P_rot_moon_HOURS, visualize_with_plot: bool = False):
    """
    This returns a planetary system, where the star is sun-like, the planet is earth-like and the moon is lunar-like.
    The mass of the submoon is set to 4.2e15kg which is the submoon test mass of Kollmeier & Raymond.
    The earth's quality factor and k2 love number are taken to be 280 and 0.3 respectively.
    [see Lainey Q Parameters document in media folder.]
    The moon's quality factor and k2 love number are taken to be 100 and 0.25 respectively, as per the caption of
    figure 1 of Kollmeier & Raymond.
    We also assume the same parameters for the submoon.

    For the sun, we use the following result by Adrian J. Barker
    (https://iopscience.iop.org/article/10.3847/2041-8213/ac5b63/pdf, see also https://arxiv.org/pdf/2307.13074 for an
    important literature overview):

    Q' = 3/2 * Q/k_2 ~ 10^7 (P_rot/(10 days))^2,

    where P_rot is the rotation period of a sun-like star (mass range 0.2 - 1.2 M_odot) about its own axis.
    To implement this formula, we set k_2_star = 1, such that the ratio Q/k can be fully represented in Q itself via

    Q = 2/3 * Q' = 2/3 * 10^7 (P_rot/(10 days))^2.

    In the DFE.'s the ratio Q/k2 appears.
    This ratio must be equal to 2/3*Q'.
    We set k=1 and the Q = 2/3*Q' thereby achieving the needed ratio.
    This means, that any calculations done that grab the stars k2-value or Q-value are wrong.
    (We don't do such calculations).

    UPDATE 23.10.24: Actually, I now set the submoon mass to 100.000 x the test mass in Kollmeier and Raymond
    because a small submoon test mass gives raise to a lot of skipped iterations due to detected stiffness, whereas
    if the submoon is massive, a lot of those points definitely reach an unstable point before getting into
    small scale oscillations.

    visualize_with_plot: bool,      Used to plot an overview of the constructed system
    P_rot_star_DAYS: float,         The rotation period of the star in days.
    P_rot_planet_HOURS: float,      The rotation period of the planet in hours.
    P_rot_moon_HOURS: float,        The rotation period of the moon in hours.

    """

    # get some parameters from solar system bodies
    sun_d = get_solar_system_bodies_data(file_to_read='constants/stars_solar_system.txt',
                                         name_of_celestial_body='Sun')

    # get some parameters from solar system bodies. The '_d' stands for '_data'.
    earth_d = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                           name_of_celestial_body='Earth')

    # get some parameters from solar system bodies
    luna_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                          name_of_celestial_body='Moon')

    submoon_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                             name_of_celestial_body='Asteroid')

    spin_frequency_sun = 1 / (3600 * (P_rot_star_DAYS*24))
    spin_frequency_earth = 1 / (3600 * P_rot_planet_HOURS)
    spin_frequency_luna = 1 / (3600 * P_rot_moon_HOURS)
    spin_frequency_submoon = None  # This should not raise any errors since should not be used according to the DFE.'s

    sun_Q = 2/3 * 10**7 * (P_rot_star_DAYS/10)**2

    sun = CelestialBody(mass=sun_d.m, density=sun_d.rho, semi_major_axis=sun_d.a,
                        spin_frequency=spin_frequency_sun, love_number=1,
                        quality_factor=sun_Q, descriptive_index="s", name="sun",
                        hierarchy_number=1, hosting_body=None)

    earth = CelestialBody(mass=earth_d.m, density=earth_d.rho, semi_major_axis=earth_d.a,
                          spin_frequency=spin_frequency_earth, love_number=0.3,
                          quality_factor=280, descriptive_index="p", name="planet",
                          hierarchy_number=2, hosting_body=sun)

    moon = CelestialBody(mass=luna_d.m, density=luna_d.rho, semi_major_axis=luna_d.a,
                         spin_frequency=spin_frequency_luna, love_number=0.25,
                         quality_factor=100, descriptive_index="m", name="moon",
                         hierarchy_number=3, hosting_body=earth)

    # submoon_mass = submoon_d.m/2.1/20
    # submoon_mass = 4.2e15  # this is the test mass used by Kollmeier & Raymond
    submoon_mass =  luna_d.m * 1/10  # 1/10th of moon mass
    # submoon_mass = 1e-200
    # submoon_mass = 1/3 * 6.39e23  # A third of mars' mass!
    submoon = CelestialBody(mass=submoon_mass, density=submoon_d.rho, semi_major_axis=submoon_d.a,
                            spin_frequency=spin_frequency_submoon, love_number=0.25,
                            quality_factor=100, descriptive_index="sm", name="submoon",
                            hierarchy_number=4, hosting_body=moon)

    if visualize_with_plot:
        """
                fig, ax = plt.subplots()

                sun_circ = Circle((0, 0), radius=1, edgecolor='black', facecolor='black', label="sun")
                sun_radius = diameter_sun/2

                earth_radius_relative = diameter_earth/2/sun_radius
                earth_distance_relative = a_earth/sun_radius
                earth_circ = Circle((earth_distance_relative, 0), radius=earth_radius_relative, edgecolor='blue', facecolor='blue',
                               label="earth")

                ganymede_radius_relative = diameter_ganymede/2/sun_radius
                ganymede_distance_relative = earth_distance_relative+a_luna/sun_radius
                ganymede_circ = Circle((ganymede_distance_relative, 0), radius=ganymede_radius_relative,
                                  edgecolor='brown', facecolor='brown', label="ganymede")

                submoon_distance_relative = ganymede_distance_relative + a_luna/sun_radius/50
                submoon_radius_relative = ganymede_radius_relative / 10
                submoon_circ = Circle((submoon_distance_relative, 0), radius=submoon_radius_relative, edgecolor='pink', facecolor='pink', label="submoon")

                ax.add_patch(sun_circ)
                ax.add_patch(earth_circ)
                ax.add_patch(ganymede_circ)
                ax.add_patch(submoon_circ)
                ax.set_aspect('equal')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-10, 250)
                ax.set_ylim(-50, 50)

                ax.vlines(0, -50, 50, color="black", lw=0.5)
                ax.vlines(earth_distance_relative, -50, 50, color="blue", lw=0.5)
                ax.vlines(ganymede_distance_relative, -50, 50, color="brown", lw=0.5)
                ax.vlines(submoon_distance_relative, -50, 50, color="pink", lw=0.5)
                ax.hlines(0, -10, 250, lw=0.5, color="black")

                ax.legend()
                plt.show()
                """

        raise ValueError("This method needs to revised.")
    return sun, earth, moon, submoon



def create_warm_jupiter_submoon_system(P_rot_star_DAYS, P_rot_planet_HOURS, P_rot_moon_HOURS):
    """
    Mimicks the first system in which an exomoon might have been found, Kepler-1625b, see paper
    https://arxiv.org/pdf/1810.02362 .

    The planet is ~3 jupiter masses (1-sigma interval according to this paper https://arxiv.org/pdf/2001.10867),
    orbiting on a likely circular orbit in a distance of 1AU a solar-mass star (https://arxiv.org/pdf/1810.02362).
    Semimajor-axis of exomoon is around 40 planetary radii, i.e. 40 x 11 earth radii. (ebenda).
    The exomoon has a radius of approximately 4 earth radii.
    The exomoon has a mean mass of 10^1.27 M_earth ~ 18.6 M_earth
    The planet has a radius of approximately 11 earth radii.

    We approximate the exomoon as Neptune-like regarding its k2/Q ratio, i.e. k2 = 0.127 and we use
    Q_neptune ~ Q_uranus ~ 5000 (Table I and last paragraph of
    https://www.sciencedirect.com/science/article/pii/001910357790015X)


    For the sun, we make the same assumptions as in `create_earth_submoon_system`.

    We approximate the planets Love Number and Quality factor as jupiter's, of which we know the mean ratio:

    k2_jup / Q_jup = 1.102e-5 (Lainey Q Parameters).

    In the DFEs, this ratio appears, thus, we set the planet's Q_jup to be 1 and its k2_jup to 1.102e-5, achieving
    the desired ratio.


    P_rot_star_DAYS: float,         The rotation period of the star in days.
    P_rot_planet_HOURS: float,      The rotation period of the planet in hours.
    P_rot_moon_HOURS: float,        The rotation period of the moon in hours.

    """

    # get some parameters from solar system bodies
    sun_d = get_solar_system_bodies_data(file_to_read='constants/stars_solar_system.txt',
                                         name_of_celestial_body='Sun')

    # get some parameters from solar system bodies. The '_d' stands for '_data'.
    planet_d = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                           name_of_celestial_body='Jupiter')

    # get some parameters from solar system bodies
    neptune_d = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                          name_of_celestial_body='Neptune')

    # get some parameters from solar system bodies
    earth_d = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                          name_of_celestial_body='Earth')

    submoon_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                             name_of_celestial_body='Asteroid')

    spin_frequency_sun = 1 / (3600 * (P_rot_star_DAYS*24))
    spin_frequency_planet = 1 / (3600 * P_rot_planet_HOURS)
    spin_frequency_luna = 1 / (3600 * P_rot_moon_HOURS)
    spin_frequency_submoon = None  # This should not raise any errors since should not be used according to the DFE.'s

    sun_Q = 2/3 * 10**7 * (P_rot_star_DAYS/10)**2
    earth_R = 6370e3

    sun = CelestialBody(mass=sun_d.m, density=sun_d.rho, semi_major_axis=sun_d.a,
                        spin_frequency=spin_frequency_sun, love_number=1,
                        quality_factor=sun_Q, descriptive_index="s", name="sun",
                        hierarchy_number=1, hosting_body=None)

    planet_density = 3*planet_d.m / (4/3 * (11*earth_R)**3 * np.pi)
    planet = CelestialBody(mass=3*planet_d.m, density=planet_density, semi_major_axis=earth_d.a,
                          spin_frequency=spin_frequency_planet, love_number=1.102e-5,
                          quality_factor=1, descriptive_index="p", name="planet",
                          hierarchy_number=2, hosting_body=sun)

    moon_density = 18.6*earth_d.m / (4/3 * (4*earth_R)**3 * np.pi)
    moon = CelestialBody(mass=18.6*earth_d.m, density=moon_density, semi_major_axis=40*11*earth_R,
                         spin_frequency=spin_frequency_luna, love_number=0.127,
                         quality_factor=5000, descriptive_index="m", name="moon",
                         hierarchy_number=3, hosting_body=planet)

    submoon_mass = 0.1 * moon.mass
    # submoon_mass = submoon_d.m/2.1/20
    # submoon_mass = 4.2e15  # this is the test mass used by Kollmeier & Raymond
    # submoon_mass = .25 * submoon_d.m  # 25% of the lunar mass
    # submoon_mass = 1e-200
    # submoon_mass = 1/3 * 6.39e23  # A third of mars' mass!
    submoon = CelestialBody(mass=submoon_mass, density=submoon_d.rho, semi_major_axis=submoon_d.a,
                            spin_frequency=spin_frequency_submoon, love_number=0.25,
                            quality_factor=100, descriptive_index="sm", name="submoon",
                            hierarchy_number=4, hosting_body=moon)


    return sun, planet, moon, submoon

