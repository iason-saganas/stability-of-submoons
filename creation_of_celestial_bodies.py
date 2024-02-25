from utilities import *
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

__all__ = ['create_toy_satellite_and_planet', 'create_toy_submoon_system']


def create_toy_satellite_and_planet():
    """
    :ToDo Write documentation.
    :ToDo Increase efficiency of data query and instantiation of 'CelestialBody' instances
    :return:
    """
    # get some parameters from solar system bodies
    (name_earth, m_earth, a_earth, diameter_earth, orbital_period_earth,
     eccentricity_earth, density_earth, rotation_period_earth, earth_love_number,
     earth_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                                          name_of_celestial_body='Earth')

    (name_luna, m_luna, a_luna, diameter_luna, orbital_period_luna,
     eccentricity_luna, density_luna, rotation_period_luna, luna_love_number,
     luna_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                                         name_of_celestial_body='Moon')

    # define spin frequencies
    spin_frequency_earth = 1 / (3600 * rotation_period_earth)
    spin_frequency_ganymede = 1 / (3600 * rotation_period_luna)

    # define first toy planet
    toy_planet = CelestialBody(mass=m_earth, density=density_earth, semi_major_axis=None,
                               spin_frequency=spin_frequency_earth, love_number=earth_love_number,
                               quality_factor=earth_quality_factor, descriptive_index="p", name="toy planet",
                               hierarchy_number=1, hosting_body=None)

    # then toy satellite
    toy_satellite = CelestialBody(mass=m_luna, density=density_luna, semi_major_axis=a_luna,
                                  spin_frequency=spin_frequency_ganymede, love_number=luna_love_number,
                                  quality_factor=luna_quality_factor, descriptive_index="s", name="toy satellite",
                                  hierarchy_number=2, hosting_body=toy_planet)

    return toy_satellite, toy_planet


def create_toy_submoon_system(visualize_with_plot: bool = False):
    """
    :ToDo Write better documentation
    Creates a submoon system consisting of sun, earth, ganymede at the same distance as earth's moon and an asteroid
    like object serving as submoon.

    sun --------- 1 AU
        ---------> earth ----- moon distance
            -----> ganymede-like object (but mass *=50 mass and radius *= 100 radius) -- 1/50th moon distance
                --> asteroid like object as submoon, 1/20 of ganymede mass, a tenth of its radius, its rotational period
                    is 2 x that of ganymede.

    :return:
    """
    # get some parameters from solar system bodies
    (name_earth, m_earth, a_earth, diameter_earth, orbital_period_earth,
     eccentricity_earth, density_earth, rotation_period_earth, earth_love_number,
     earth_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                                          name_of_celestial_body='Earth')

    # get some parameters from solar system bodies
    (name_sun, m_sun, a_sun, diameter_sun, orbital_period_sun,
     eccentricity_sun, density_sun, rotation_period_sun, sun_love_number,
     sun_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/stars_solar_system.txt',
                                                        name_of_celestial_body='Sun')

    # get some parameters from solar system bodies
    (name_ganymede, m_ganymede, a_ganymede, diameter_ganymede, orbital_period_ganymede,
     eccentricity_ganymede, density_ganymede, rotation_period_ganymede, ganymede_love_number,
     ganymede_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                                             name_of_celestial_body='Ganymede')

    (name_luna, m_luna, a_luna, diameter_luna, orbital_period_luna,
     eccentricity_luna, density_luna, rotation_period_luna, luna_love_number,
     luna_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                                         name_of_celestial_body='Moon')

    (name_submoon, m_submoon, a_submoon, diameter_submoon, orbital_period_submoon,
     eccentricity_submoon, density_submoon, rotation_period_submoon, submoon_love_number,
     submoon_quality_factor) = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                                            name_of_celestial_body='Asteroid')

    spin_frequency_sun = 1 / (3600 * rotation_period_sun)
    spin_frequency_earth = 1 / (3600 * rotation_period_earth)
    spin_frequency_ganymede = 1 / (3600 * rotation_period_ganymede)
    spin_frequency_submoon = 1 / (3600 * rotation_period_submoon)

    sun = CelestialBody(mass=m_sun, density=density_sun, semi_major_axis=a_sun,
                        spin_frequency=spin_frequency_sun, love_number=sun_love_number,
                        quality_factor=sun_quality_factor, descriptive_index="s", name="sun",
                        hierarchy_number=1, hosting_body=None)

    earth = CelestialBody(mass=m_earth, density=density_earth, semi_major_axis=a_earth,
                          spin_frequency=spin_frequency_earth, love_number=earth_love_number,
                          quality_factor=earth_quality_factor, descriptive_index="p", name="planet",
                          hierarchy_number=2, hosting_body=sun)

    moon = CelestialBody(mass=m_ganymede, density=density_ganymede, semi_major_axis=a_luna,
                         spin_frequency=spin_frequency_ganymede, love_number=ganymede_love_number,
                         quality_factor=ganymede_quality_factor, descriptive_index="m", name="moon",
                         hierarchy_number=3, hosting_body=earth)

    submoon = CelestialBody(mass=m_submoon, density=density_submoon, semi_major_axis=a_submoon,
                            spin_frequency=spin_frequency_submoon, love_number=submoon_love_number,
                            quality_factor=submoon_quality_factor, descriptive_index="sm", name="submoon",
                            hierarchy_number=4, hosting_body=moon)

    if visualize_with_plot:
        fig, ax = plt.subplots()

        sun = Circle((0, 0), radius=1, edgecolor='black', facecolor='black', label="sun")
        sun_radius = diameter_sun/2

        earth_radius_relative = diameter_earth/2/sun_radius
        earth_distance_relative = a_earth/sun_radius
        earth = Circle((earth_distance_relative, 0), radius=earth_radius_relative, edgecolor='blue', facecolor='blue',
                       label="earth")

        ganymede_radius_relative = diameter_ganymede/2/sun_radius
        ganymede_distance_relative = earth_distance_relative+a_luna/sun_radius
        ganymede = Circle((ganymede_distance_relative, 0), radius=ganymede_radius_relative,
                          edgecolor='brown', facecolor='brown', label="ganymede")

        submoon_distance_relative = ganymede_distance_relative + a_luna/sun_radius/50
        submoon_radius_relative = ganymede_radius_relative / 10
        submoon = Circle((submoon_distance_relative, 0), radius=submoon_radius_relative, edgecolor='pink', facecolor='pink', label="submoon")

        ax.add_patch(sun)
        ax.add_patch(earth)
        ax.add_patch(ganymede)
        ax.add_patch(submoon)
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

    return sun, earth, moon, submoon

