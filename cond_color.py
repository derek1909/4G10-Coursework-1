import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

black_c = lambda c: LinearSegmentedColormap.from_list('BlkGrn', [(0, 0, 0), c], N=256)


def get_colors(xs, ys, alt_colors=False):
    """
    returns a list of colors (that can be passed as the optional "color" or "c" input arguments to Matplotlib plotting functions)
    based on the values in the coordinate lists (or 1D array) xs and ys. More specifically, colors are based on the
    projected location along a direction with the widest spread of points.
    :param xs, ys: two vectors (lists or 1D arrays of the same length) containing the x and y coordinates of a set of points
    :param alt_colors: if True, the green and red color poles (for negative and positive values) are switched to cyan and magenta.
    :return:
    colors: a list (with same length as xs) of colors corresponding to coorinates along the maximum-spread direction:
    small values are closer to black, large positive values closer to red, and large negative values closer to green.
    The elements of "colors" can be passed as the optional "color" or "c" input argument to Matplotlib plotting functions.
    """
    xys = np.array([xs, ys])
    u, _, _ = np.linalg.svd(xys)
    normalize = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
    xs = normalize(u[:,0].T @ xys)
    if alt_colors:
        pos_cmap = black_c((1, 0, 1))
        neg_cmap = black_c((0, 1, 1))
    else:
        pos_cmap = black_c((1, 0, 0))
        neg_cmap = black_c((0, 1, 0))

    colors = []
    for x in xs:
        if x < 0:
            colors.append(neg_cmap(-x))
        else:
            colors.append(pos_cmap(+x))

    return colors


def plot_start(xs, ys, colors, markersize=500, ax=None, alpha=1):
    """
    Puts round markers on the starting point of trajectories
    :param xs: x-coordinates of the initial point of trajectories
    :param ys: y-coordinates of the initial point of trajectories
    :param colors: colors for different conditions obtained using the get_colors function
    :param markersize: size of the markers
    :param ax: axis on which to plot (optional)
    """
    if ax is None:
        plt.scatter(xs, ys, s=markersize, color=colors, marker=".", edgecolors="k", alpha=alpha)
    else:
        ax.scatter(xs, ys, s=markersize, color=colors, marker=".", edgecolors="k", alpha=alpha)


def plot_end(xs, ys, colors, markersize=100, ax=None, alpha=1):
    """
    Puts diamond-shaped markers on the end point of trajectories
    :param xs: x-coordinates of the final point of trajectories
    :param ys: y-coordinates of the final point of trajectories
    :param colors: colors for different conditions obtained using the get_colors function
    :param markersize: size of the markers
    :param ax: axis on which to plot (optional)
    """
    if ax is None:
        plt.scatter(xs, ys, s=markersize, color=colors, marker="D", edgecolors="k", alpha=alpha)
    else:
        ax.scatter(xs, ys, s=markersize, color=colors, marker="D", edgecolors="k", alpha=alpha)

def compute_Pfr(eigenvalues, eigenvectors, plane):
    eigenvalues_imag_parts = np.imag(eigenvalues)
    v = eigenvectors[:, plane] # the largest angular speed.
    # w = eigenvalues_imag_parts[2*plane]
    # print('Angular speed: ', w)

    real_part = np.real(v)
    imag_part = np.imag(v)
    print('Verify orthogonal: ', real_part.T @ imag_part)

    u_real = real_part / np.linalg.norm(real_part)
    u_imag = imag_part / np.linalg.norm(imag_part)

    Pfr = np.array([u_real,u_imag]) 
    return Pfr

def plot_Z_projection(Z_2dplane, ax=None, title=None, alpha=1, alt_color=False):

    # Extract the initial points at -150 ms for each condition
    xstart = Z_2dplane[0, :, 0]  # x-coordinates at -150 ms
    ystart = Z_2dplane[1, :, 0]  # y-coordinates at -150 ms
    xend = Z_2dplane[0, :, -1]  # x-coordinates at -150 ms
    yend = Z_2dplane[1, :, -1]  # y-coordinates at -150 ms
    colors = get_colors(xstart, ystart,alt_color)

    # Plot Z_3d_vis for all conditions over all time points
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Create a new figure and axis if none is provided

    for c in range(Z_2dplane.shape[1]):
        ax.plot(Z_2dplane[0, c, :], Z_2dplane[1, c, :], color=colors[c], alpha=alpha)

    plot_start(xstart, ystart, colors, markersize=50, ax=ax, alpha=alpha)
    plot_end(xend, yend, colors, markersize=30, ax=ax, alpha=alpha)

    # ax.set_xlabel(f'FR{plane+1} Plane')
    # ax.set_ylabel("Axis 2")
    if title is not None:
        ax.set_title(title)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    # ax.grid()
    ax.axis("equal")  # Ensures equal scaling on both axes

    if ax is None:
        plt.tight_layout()  # Adjust layout only if we created the figure
        plt.show()
    
