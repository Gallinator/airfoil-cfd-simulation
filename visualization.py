import matplotlib
import matplotlib.pyplot as plt
import numpy as np

COLORMAP = 'jet'


def plot_airfoil(landmarks: np.ndarray, grid_x: np.array, grid_y: np.array, momentum_x: np.array, momentum_y: np.array):
    # Use the momentum magnitude as color
    color = np.sqrt(np.square(momentum_y) + np.square(momentum_x))
    vmin, vmax = np.min(color), np.max(color)

    fig, ax = plt.subplots(1)
    ax.set_title("Flow momentum")
    ax.fill(landmarks[:, 0], landmarks[:, 1], color='black')
    ax.quiver(grid_x, grid_y, momentum_x, momentum_y, color,
              scale_units='xy',
              units='xy',
              scale=6,
              headwidth=2,
              headlength=4,
              headaxislength=4,
              width=0.002,
              cmap='jet')
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=COLORMAP),
                 orientation='vertical',
                 label='Momentum',
                 ax=ax)
    plt.show()
