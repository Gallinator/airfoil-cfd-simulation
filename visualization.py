import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from loss_tracker import LossTracker

COLORMAP = 'jet'
plt.style.use('dark_background')


def plot_training_history(loss_tracker: LossTracker):
    fig, ax = plt.subplots(1)

    ax.set_title('Training loss')
    ax.set_xlabel('Epoch')
    for l, h in loss_tracker.loss_history.items():
        x = np.arange(len(h))
        ax.plot(x, h, label=l)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    plt.tight_layout()


def plot_airfoil(alpha, landmarks: np.ndarray, grid_x: np.array, grid_y: np.array, u: np.array, v: np.array,
                 p: np.array):
    color = np.sqrt(np.square(v) + np.square(u))
    vmin, vmax = np.min(color), np.max(color)
    grid_edge_size = int(math.sqrt(len(grid_x)))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Airflow simulation, AoA={int(alpha)}', fontsize=16)

    ax_v, ax_p = axs
    ax_v.set_title(f"Flow velocity")
    ax_v.fill(landmarks[:, 0], landmarks[:, 1], color='grey')
    ax_v.quiver(grid_x, grid_y, u, v, color,
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
                 label='Velocity',
                 ax=ax_v)

    ax_stream = axs[2]
    ax_stream.set_title(f"Airflow")
    ax_stream.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    ax_stream.streamplot(grid_x.reshape(grid_edge_size, -1).T,
                         grid_y.reshape(grid_edge_size, -1).T,
                         u.reshape(grid_edge_size, -1).T,
                         v.reshape(grid_edge_size, -1).T,
                         color=color.reshape(grid_edge_size, -1).T,
                         broken_streamlines=False, arrowsize=0, density=2, cmap='jet')
    ax_stream.set_aspect('equal')

    plt.tight_layout()
    plt.show()
