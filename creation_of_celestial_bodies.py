from utilities import *
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

__all__ = ['create_toy_satellite_and_planet', 'create_submoon_system']


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


def create_submoon_system(visualize_with_plot: bool = False):
    """
    :ToDo Write documentation
    """

    # get some parameters from solar system bodies
    sun_d = get_solar_system_bodies_data(file_to_read='constants/stars_solar_system.txt',
                                         name_of_celestial_body='Sun')

    # get some parameters from solar system bodies. The '_d' stands for '_data'.
    earth_d = get_solar_system_bodies_data(file_to_read='constants/planets_solar_system.txt',
                                           name_of_celestial_body='Earth')

    # get some parameters from solar system bodies
    ganymede_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                              name_of_celestial_body='Ganymede')

    luna_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                          name_of_celestial_body='Moon')

    submoon_d = get_solar_system_bodies_data(file_to_read='constants/moons_solar_system.txt',
                                             name_of_celestial_body='Asteroid')

    spin_frequency_sun = 1 / (3600 * sun_d.T_rotation_hours)
    spin_frequency_earth = 1 / (3600 * earth_d.T_rotation_hours)
    spin_frequency_luna = 1 / (3600 * luna_d.T_rotation_hours)
    spin_frequency_ganymede = 1 / (3600 * ganymede_d.T_rotation_hours)
    spin_frequency_submoon = 1 / (3600 * submoon_d.T_rotation_hours)

    sun = CelestialBody(mass=sun_d.m, density=sun_d.rho, semi_major_axis=sun_d.a,
                        spin_frequency=spin_frequency_sun, love_number=sun_d.k,
                        quality_factor=sun_d.Q, descriptive_index="s", name="sun",
                        hierarchy_number=1, hosting_body=None)

    earth = CelestialBody(mass=earth_d.m, density=earth_d.rho, semi_major_axis=earth_d.a,
                          spin_frequency=spin_frequency_earth, love_number=earth_d.k,
                          quality_factor=earth_d.Q, descriptive_index="p", name="planet",
                          hierarchy_number=2, hosting_body=sun)

    moon = CelestialBody(mass=luna_d.m, density=luna_d.rho, semi_major_axis=luna_d.a,
                         spin_frequency=spin_frequency_luna, love_number=luna_d.k,
                         quality_factor=luna_d.Q, descriptive_index="m", name="moon",
                         hierarchy_number=3, hosting_body=earth)

    submoon = CelestialBody(mass=submoon_d.m/2.1*0, density=submoon_d.rho, semi_major_axis=submoon_d.a*5,
                            spin_frequency=spin_frequency_submoon, love_number=submoon_d.k,
                            quality_factor=submoon_d.Q, descriptive_index="sm", name="submoon",
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

