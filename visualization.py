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

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
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

    ax_p.set_title(f"Flow dynamic pressure")
    ax_p.fill(landmarks[:, 0], landmarks[:, 1], color='grey')
    ax_p.scatter(grid_x, grid_y, c=p)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(p), np.max(p)), cmap=COLORMAP),
                 orientation='vertical',
                 label='Pressure',
                 ax=ax_p)
    plt.show()
