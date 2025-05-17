from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


def plot_3d_voxels_initial_states(data_solved, data_unsolved, save=False, filename=None, plot=True,
                                  system = "earth_like"):
    """

    Each data object has the following structure: [X, Y, Z, color_values], where
    X, Y, Z and values are all 1D np.array where the elements at each index correspond to a one datapoint,
    i.e. location and transparency value (alpha channel).

    :param data_solved:         Data array of those iterations that were completed successfully without
                                premature interruption due to stiffness detection.
    :param data_unsolved:       !data_solved
    :return:
    """

    X_sol, Y_sol, Z_sol, values_sol = data_solved
    X_unsol, Y_unsol, Z_unsol, values_unsol = data_unsolved

    # Create figure and grid
    fig = plt.figure(figsize=(12, 8))
    ax_main = plt.subplot(111, projection='3d')
    # ax_main.view_init(30, 340)
    ax_main.view_init(14, -42)

    X_merge = np.concatenate((X_sol, X_unsol)).flatten()
    Y_merge = np.concatenate((Y_sol, Y_unsol)).flatten()
    Z_merge = np.concatenate((Z_sol, Z_unsol)).flatten()
    #
    # draw_3d_box(ax=ax_main, x_min=min(X_merge), x_max=max(X_merge), y_min=min(Y_merge), y_max=max(Y_merge),
    #             z_min=min(Z_merge), z_max=max(Z_merge))

    # Set labels
    ax_main.set_xlabel(r'$a_{\mathrm{p}} [\mathrm{AU}]$', fontsize=20, rotation=150, labelpad=20)
    ax_main.set_ylabel(r'$a_{\mathrm{m}} [R_{\mathrm{p}}]$', fontsize=20, labelpad=20)
    ax_main.set_zlabel(r'$a_{\mathrm{sm}} [R_{\mathrm{m}}]$', fontsize=20, rotation=90, labelpad=20)

    # Custom color map
    N = 256
    color_array = plt.get_cmap('Blues')(range(N))
    # change alpha values
    color_array[:, 3] = np.linspace(0, 1, N)  # Let the alpha channel linearly increase, importantly,
    # let the value 0 correspond to the alpha value 0 such that unstable regions are transparent
    #color_array[:, 2] = np.linspace(0, 1, N)  # Let the green channel linearly increase
    #color_array[:, 1] = np.linspace(0, 1, N)  # Let the blue channel linearly increase
    #color_array[:, 0] = np.linspace(0, 1, N)  # Let the red channel linearly increase

    # seismic_r is what barbara suggested
    seismic = "seismic_r"
    fading_blue = LinearSegmentedColormap.from_list(name="myColorMap", colors=color_array)  # Custom color map

    p = ax_main.scatter(X_sol, Y_sol, Z_sol, s=30, c=values_sol, cmap=fading_blue)  # Plot properly solved points
    q = ax_main.scatter(X_unsol, Y_unsol, Z_unsol, s=30, marker='x', color="black")  # Plot stiff, unsolved points

    earth_distance = 1 # AU
    luna_distance = 60  # in earth radii

    kepler1625b_planet_distance = 1  # AU
    hypothesized_moon_distance = 40

    if system == "earth_like":
        print("The system is earth like, plotting the real earth system.")
        descr = "Earth"
        real_sys_x = earth_distance
        real_sys_y = luna_distance
    else:
        print("The system is warm jupiter like, plotting the Kepler1625b system.")
        descr = "Kepler1625b"
        real_sys_x = kepler1625b_planet_distance
        real_sys_y = hypothesized_moon_distance

    # REAL SYSTEM
    draw_beam(ax_main, x_center=real_sys_x, y_center=real_sys_y, z_min=0, z_max=max(Z_merge),
              step_x = 0.05, step_y=5, alpha=1, color="lightgreen")

    ax_main.set_zlim(0, max(Z_sol))  # Adjust max_z_value as needed

    ax_main.text(x=real_sys_x, y=real_sys_y, z=max(Z_merge) + 3, s=descr, color='darkgreen',
                 fontsize=16, ha='center', va='bottom')

    # cbar = fig.colorbar(p, ax=ax_main, pad=0.2, fraction=0.03)  # Add colorbar
    # cbar.set_label(r"Lifetime $[4.5\mathrm{Gyr}]$", fontsize=22, labelpad=22)
    # plt.title("Initial states, "+r"$m_{\mathrm{sm}}=10^{15}\mathrm{kg}$", pad=1)
    plt.tight_layout()
    if save:
        if filename is None:
            raise ValueError("Provide filename")
        plt.savefig("data_storage/figures/"+filename, bbox_inches='tight', pad_inches=0.1)
    if plot:
        plt.show()
    else:
        plt.clf()


def draw_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max):
    # Define the 8 corners of the box
    corners = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ]

    # Define the 12 edges of the box, each edge is a pair of corners
    edges = [
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]],
        [corners[4], corners[5]],
        [corners[5], corners[6]],
        [corners[6], corners[7]],
        [corners[7], corners[4]],
        [corners[0], corners[4]],
        [corners[1], corners[5]],
        [corners[2], corners[6]],
        [corners[3], corners[7]]
    ]

    # Plot each edge
    for edge in edges:
        x_values = [edge[0][0], edge[1][0]]
        y_values = [edge[0][1], edge[1][1]]
        z_values = [edge[0][2], edge[1][2]]
        ax.plot(x_values, y_values, z_values, color='black')


def draw_1d_line(ax, x_val, y_val, z_min, z_max):
    """
    Draw a 1D line at fixed x and y values, spanning from z_min to z_max.

    :param ax: The 3D axis on which to plot the line.
    :param x_val: The fixed x value for the line.
    :param y_val: The fixed y value for the line.
    :param z_min: The minimum z value.
    :param z_max: The maximum z value.
    """
    # Create the line from z_min to z_max at the given x and y
    ax.plot([x_val, x_val], [y_val, y_val], [z_min, z_max], color='black', ls="-", markersize=5, lw=8)


def draw_beam(ax, x_center, y_center, z_min, z_max, step_x, step_y, color="lightgreen", alpha=0.3):
    """
    Draws a beam with four vertical support lines at the corners and connects them with panels:
    - Bottom panel at z_min (blue, transparent)
    - Top panel at z_max (blue, transparent)
    - Side panels (red, transparent) for the front, back, left, and right walls.

    :param ax: The 3D axis to draw on.
    :param x_center: Center x-coordinate.
    :param y_center: Center y-coordinate.
    :param z_min: Bottom z value.
    :param z_max: Top z value.
    :param step_x: Horizontal half-width (x-direction).
    :param step_y: Vertical half-width (y-direction).
    """
    # Define the 4 corner points for the beam (for vertical support lines)
    corners = [
        (x_center - step_x, y_center - step_y),
        (x_center + step_x, y_center - step_y),
        (x_center + step_x, y_center + step_y),
        (x_center - step_x, y_center + step_y)
    ]

    # Plot the 4 vertical support lines at the corners
    # for (x, y) in corners:
    #     ax.plot([x, x], [y, y], [z_min, z_max], color='b')

    # ----- Bottom Panel -----
    # Create a 2x2 grid for bottom surface at z_min
    x_bottom, y_bottom = np.meshgrid([x_center - step_x, x_center + step_x],
                                     [y_center - step_y, y_center + step_y])
    z_bottom = np.full_like(x_bottom, z_min)
    ax.plot_surface(x_bottom, y_bottom, z_bottom, color=color, alpha=alpha)

    # ----- Top Panel -----
    # Create a 2x2 grid for top surface at z_max
    x_top, y_top = np.meshgrid([x_center - step_x, x_center + step_x],
                               [y_center - step_y, y_center + step_y])
    z_top = np.full_like(x_top, z_max)
    ax.plot_surface(x_top, y_top, z_top, color=color, alpha=alpha)

    # Define helper variables for side panels:
    # x_vals and y_vals for the edges
    x_vals = [x_center - step_x, x_center + step_x]
    y_vals = [y_center - step_y, y_center + step_y]
    z_vals = [z_min, z_max]  # vertical range

    # ----- Back Panel (x constant = x_center + step_x) -----
    x_back = np.full((2, 2), x_center + step_x)
    y_back, z_back = np.meshgrid(y_vals, z_vals)
    ax.plot_surface(x_back, y_back, z_back, color=color, alpha=alpha)

    # ----- Front Panel (x constant = x_center - step_x) -----
    x_front = np.full((2, 2), x_center - step_x)
    y_front, z_front = np.meshgrid(y_vals, z_vals)
    ax.plot_surface(x_front, y_front, z_front, color=color, alpha=alpha)

    # ----- Left Panel (y constant = y_center - step_y) -----
    y_left = np.full((2, 2), y_center - step_y)
    x_left, z_left = np.meshgrid(x_vals, z_vals)
    ax.plot_surface(x_left, y_left, z_left, color=color, alpha=alpha)

    # ----- Right Panel (y constant = y_center + step_y) -----
    y_right = np.full((2, 2), y_center + step_y)
    x_right, z_right = np.meshgrid(x_vals, z_vals)
    ax.plot_surface(x_right, y_right, z_right, color=color, alpha=alpha)

def plot_3d_voxels_final_states(data, save=False, filename=None, plot=True, system="earth_like"):
    """

    A less complex copy of `plot_3d_voxels_initial_states`.

    :param data:         Data array of those iterations that were completed successfully without premature interruption
                         due to bad initial values.
                         Might contain iterations that were detected as stiff.
    :return:
    """

    X_sol, Y_sol, Z_sol, values_sol = data

    copy_X, copy_Y, copy_Z, copy_val_sols= [np.array(arr) for arr in [X_sol, Y_sol, Z_sol, values_sol]]
    idcs = np.where((copy_X < 1.3) & (copy_val_sols > 0.4) & (copy_Y < 70))[0]
    for idx in idcs:
        print("At ", X_sol[idx], Y_sol[idx], Z_sol[idx], " we have f  = ", values_sol[idx])

    # Create figure and grid
    fig = plt.figure(figsize=(12, 8))
    ax_main = plt.subplot(111, projection='3d')
    # ax_main.view_init(30, 340)
    ax_main.view_init(27, -66)



    # draw_3d_box(ax=ax_main, x_min=min(X_sol), x_max=max(X_sol), y_min=min(Y_sol), y_max=max(Y_sol),
    #             z_min=min(Z_sol), z_max=max(Z_sol))

    # Set labels
    ax_main.set_xlabel(r'$a_{\mathrm{p}} [\mathrm{AU}]$', fontsize=20, rotation=150, labelpad=20)
    ax_main.set_ylabel(r'$a_{\mathrm{m}} [R_{\mathrm{p}}]$', fontsize=20, labelpad=20)
    ax_main.set_zlabel(r'$a_{\mathrm{sm}} [R_{\mathrm{m}}]$', fontsize=20, rotation=90, labelpad=20)

    # Custom color map
    N = 256
    color_array = plt.get_cmap('Blues')(range(N))
    # change alpha values
    color_array[:, 3] = np.linspace(0, 1, N)  # Let the alpha channel linearly increase, importantly,
    # let the value 0 correspond to the alpha value 0 such that unstable regions are transparent
    #color_array[:, 2] = np.linspace(0, 1, N)  # Let the green channel linearly increase
    #color_array[:, 1] = np.linspace(0, 1, N)  # Let the blue channel linearly increase
    #color_array[:, 0] = np.linspace(0, 1, N)  # Let the red channel linearly increase

    # seismic_r is what barbara suggested
    # seismic = "seismic_r"
    fading_blue = LinearSegmentedColormap.from_list(name="myColorMap", colors=color_array)  # Custom color map

    p = ax_main.scatter(X_sol, Y_sol, Z_sol, s=30, c=values_sol, cmap=fading_blue)  # Plot properly solved points

    earth_distance = 1  # AU
    luna_distance = 60  # in earth radii

    kepler1625b_planet_distance = 1  # AU
    hypothesized_moon_distance = 40

    if system == "earth_like":
        print("The system is earth like, plotting the real earth system.")
        descr = "Earth"
        real_sys_x = earth_distance
        real_sys_y = luna_distance
    else:
        print("The system is warm jupiter like, plotting real Kepler1625b. ")
        descr = "Kepler1625b"
        real_sys_x = kepler1625b_planet_distance
        real_sys_y = hypothesized_moon_distance

    # REAL EARTH SYSTEM
    draw_beam(ax_main, x_center=real_sys_x, y_center=real_sys_y, z_min=0, z_max=max(Z_sol),
              step_x=0.05, step_y=5, alpha=1, color="lightgreen")

    ax_main.set_zlim(0, max(Z_sol))  # Adjust max_z_value as needed



    ax_main.text(x=real_sys_x, y=real_sys_y, z=max(Z_sol) + 3, s=descr, color='darkgreen',
                 fontsize=16, ha='center', va='bottom')

    # submoon_mass_text = r"$m_{\mathrm{sm}}=10^{15}\mathrm{kg}$"
    # ax_main.text(x=3, y=-150, z=-5, s=submoon_mass_text, color='black',
    #              fontsize=22, ha='center', va='bottom')

    cbar = fig.colorbar(p, ax=ax_main, pad=0.1, fraction=0.03)  # Add colorbar
    cbar.set_label(r"Lifetime $[4.5\mathrm{Gyr}]$", fontsize=22, labelpad=22)
    # plt.title("Evolved states, "+r"$m_{\mathrm{sm}}=10^{15}\mathrm{kg}$")
    # s = "Evolved states, "+r"$m_{\mathrm{sm}}=10^{15}\mathrm{kg}$"
    # ax_main.text(x=0, y=max(Y_sol), z=max(Z_sol)+10, s=s, color='black', fontsize=25, ha='center', va='bottom')
    plt.tight_layout()
    if save:
        if filename is None:
            raise ValueError("Provide filename")
        plt.savefig("data_storage/figures/"+filename, bbox_inches='tight', pad_inches=0.1)
    if plot:
        plt.show()
    else:
        plt.clf()


