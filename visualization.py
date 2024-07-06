import math
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


def plot_airfoil(alpha, landmarks: np.ndarray,
                 mask: np.ndarray,
                 grid_x: np.ndarray,
                 grid_y: np.ndarray,
                 u: np.ndarray,
                 v: np.ndarray,
                 rho: np.ndarray):
    color = np.sqrt(np.square(v) + np.square(u))
    grid_edge_size = int(math.sqrt(len(grid_x)))
    airfoil_mask = mask != 1.0

    fig, axs = plt.subplots(1, 3, figsize=(36, 12), layout='constrained')
    fig.suptitle(f'Airflow simulation, AoA={int(alpha)}', fontsize=16)

    ax_v, ax_rho, ax_stream = axs
    ax_v.set_title(f"Flow velocity")
    ax_v.fill(landmarks[:, 0], landmarks[:, 1], color='grey')
    v_plot = ax_v.quiver(grid_x[airfoil_mask],
                         grid_y[airfoil_mask],
                         u[airfoil_mask],
                         v[airfoil_mask],
                         color[airfoil_mask],
                         scale_units='xy',
                         units='xy',
                         scale=6,
                         headwidth=2,
                         headlength=4,
                         headaxislength=4,
                         width=0.002,
                         cmap='jet')
    fig.colorbar(v_plot, ax=ax_v)
    ax_v.set_aspect('equal')

    ax_rho.set_title(f"Flow density")
    ax_rho.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    rho_plot = ax_rho.scatter(grid_x[airfoil_mask],
                              grid_y[airfoil_mask],
                              c=rho[airfoil_mask],
                              s=5)
    fig.colorbar(rho_plot, ax=ax_rho)
    ax_rho.set_aspect('equal')

    ax_stream.set_title(f"Airflow")
    ax_stream.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    ax_stream.streamplot(grid_x.reshape(grid_edge_size, -1).T,
                         grid_y.reshape(grid_edge_size, -1).T,
                         u.reshape(grid_edge_size, -1).T,
                         v.reshape(grid_edge_size, -1).T,
                         color=color.reshape(grid_edge_size, -1).T,
                         broken_streamlines=False, arrowsize=0, density=2, cmap='jet')
    ax_stream.set_aspect('equal')

    plt.show()
