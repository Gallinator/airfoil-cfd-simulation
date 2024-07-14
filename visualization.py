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
                 rho: np.ndarray,
                 energy: np.ndarray):
    airfoil_mask = mask != 1.0

    fig = plt.figure(figsize=(36, 12), layout='constrained')
    fig.suptitle(f'Airflow simulation, AoA={int(alpha)}', fontsize=16)

    ax_v, ax_r, ax_e = fig.subplots(1, 3)

    ax_v.set_title(f"Flow velocity")
    velocity = np.sqrt(np.square(v) + np.square(u))
    ax_v.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    v_plot = ax_v.scatter(grid_x[airfoil_mask], grid_y[airfoil_mask], c=velocity[airfoil_mask], s=3)
    fig.colorbar(v_plot, ax=ax_v)
    ax_v.set_aspect('equal')

    ax_r.set_title(f"Flow density")
    ax_r.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    r_plot = ax_r.scatter(grid_x[airfoil_mask], grid_y[airfoil_mask], c=rho[airfoil_mask], s=3)
    fig.colorbar(r_plot, ax=ax_r)
    ax_r.set_aspect('equal')

    ax_e.set_title(f"Flow energy")
    ax_e.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    e_plot = ax_e.scatter(grid_x[airfoil_mask], grid_y[airfoil_mask], c=energy[airfoil_mask], s=3)
    fig.colorbar(e_plot, ax=ax_e)
    ax_e.set_aspect('equal')
    plt.ion()
    plt.show()

    fig = plt.figure(figsize=(18, 18), layout='constrained')
    ax_stream = fig.subplots(1, 1)
    ax_stream.set_title(f"Airflow")
    ax_stream.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    ax_stream.streamplot(grid_x.T, grid_y.T, u.T, v.T, color=velocity.T,
                         broken_streamlines=False, arrowsize=0, density=3, cmap='jet')
    ax_stream.set_aspect('equal')

    plt.ioff()
    plt.show()
