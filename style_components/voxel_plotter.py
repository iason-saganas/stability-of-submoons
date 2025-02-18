from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


def plot_3d_voxels_initial_states(data_solved, data_unsolved, save=False, filename=None, plot=True):
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
    gs = gridspec.GridSpec(5, 3, height_ratios=[1] * 5, width_ratios=[1,3,1])
    ax_main = plt.subplot(gs[:, 1], projection='3d')
    ax_main.view_init(30, 340)

    X_merge = np.concatenate((X_sol, X_unsol)).flatten()
    Y_merge = np.concatenate((Y_sol, Y_unsol)).flatten()
    Z_merge = np.concatenate((Z_sol, Z_unsol)).flatten()
    draw_3d_box(ax=ax_main, x_min=min(X_merge), x_max=max(X_merge), y_min=min(Y_merge), y_max=max(Y_merge),
                z_min=min(Z_merge), z_max=max(Z_merge))

    # Set labels
    ax_main.set_xlabel(r'$a_p [\mathrm{AU}]$', fontsize=20, rotation=150, labelpad=20)
    ax_main.set_ylabel(r'$a_m [\mathrm{LU}]$', fontsize=20, labelpad=20)
    ax_main.set_zlabel(r'$a_{sm} [\mathrm{SLU}]$', fontsize=20, rotation=90, labelpad=20)

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

    fig.colorbar(p, ax=ax_main, pad=0.3, fraction=0.03)  # Add colorbar
    plt.title("Starting conditions stability space")
    if save:
        if filename is None:
            raise ValueError("Provide filename")
        plt.savefig("data_storage/figures/"+filename)
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


def plot_3d_voxels_final_states(data, save=False, filename=None, plot=True):
    """

    A less complex copy of `plot_3d_voxels_initial_states`.

    :param data:         Data array of those iterations that were completed successfully without premature interruption
                         due to bad initial values.
                         Might contain iterations that were detected as stiff.
    :return:
    """

    X_sol, Y_sol, Z_sol, values_sol = data

    # Create figure and grid
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1] * 5, width_ratios=[1,3,1])
    ax_main = plt.subplot(gs[:, 1], projection='3d')
    ax_main.view_init(30, 340)

    draw_3d_box(ax=ax_main, x_min=min(X_sol), x_max=max(X_sol), y_min=min(Y_sol), y_max=max(Y_sol),
                z_min=min(Z_sol), z_max=max(Z_sol))

    # Set labels
    ax_main.set_xlabel(r'$a_p [\mathrm{AU}]$', fontsize=20, rotation=150, labelpad=20)
    ax_main.set_ylabel(r'$a_m [\mathrm{LU}]$', fontsize=20, labelpad=20)
    ax_main.set_zlabel(r'$a_{sm} [\mathrm{SLU}]$', fontsize=20, rotation=90, labelpad=20)

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

    fig.colorbar(p, ax=ax_main, pad=0.3, fraction=0.03)  # Add colorbar
    plt.title("Final states stability space")
    if save:
        if filename is None:
            raise ValueError("Provide filename")
        plt.savefig("data_storage/figures/"+filename)
    if plot:
        plt.show()
    else:
        plt.clf()


