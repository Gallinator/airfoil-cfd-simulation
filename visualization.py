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


def plot_raw_data(alpha: int, grid_x, grid_y, landmarks, u, v, rho, energy, omega, cl, cd, cm):
    fig = plt.figure(figsize=(36, 12), layout='constrained')
    fig.suptitle(f'AoA={int(alpha)}, $C_d$ = {cd:.3f} $C_l$ = {cl:.3f} $C_m$ = {cm:.3f}', fontsize=16)
    ax_u, ax_v, ax_r, ax_o, ax0, ax_e = fig.subplots(2, 3).flatten()
    ax_u.set_title('Rho U')
    ax_u.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    u_plot = ax_u.scatter(grid_x, grid_y, c=u, s=1)
    fig.colorbar(u_plot, ax=ax_u)
    ax_u.set_aspect('equal')

    ax_v.set_title('Rho V')
    ax_v.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    v_plot = ax_v.scatter(grid_x, grid_y, c=v, s=1)
    fig.colorbar(v_plot, ax=ax_v)
    ax_v.set_aspect('equal')

    ax_r.set_title('Rho')
    ax_r.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    r_plot = ax_r.scatter(grid_x, grid_y, c=rho, s=1)
    fig.colorbar(r_plot, ax=ax_r)
    ax_r.set_aspect('equal')

    ax_e.set_title('Energy')
    ax_e.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    e_plot = ax_e.scatter(grid_x, grid_y, c=energy, s=1)
    fig.colorbar(e_plot, ax=ax_e)
    ax_e.set_aspect('equal')

    ax_o.set_title('Omega')
    ax_o.fill(landmarks[:, 0], landmarks[:, 1], color='grey', zorder=10)
    o_plot = ax_o.scatter(grid_x, grid_y, c=omega, s=1)
    fig.colorbar(o_plot, ax=ax_o)
    ax_o.set_aspect('equal')

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()


def plot_airfoil(alpha, landmarks: np.ndarray,
                 mask: np.ndarray,
                 grid_x: np.ndarray,
                 grid_y: np.ndarray,
                 u: np.ndarray,
                 v: np.ndarray,
                 rho: np.ndarray,
                 energy: np.ndarray
                 , cl: float, cd: float, cm: float):
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
    u[mask == 1] = 0
    v[mask == 1] = 0
    ax_stream.streamplot(grid_x.T, grid_y.T, u.T, v.T, color=velocity.T,
                         broken_streamlines=False, arrowsize=0, density=3, cmap='jet')
    coefs_text = f'$C_d$ = {cd:.3f}\n$C_l$ = {cl:.3f}\n$C_m$ = {cm:.3f}'
    box_props = dict(boxstyle='round', alpha=0.9)
    ax_stream.text(0.05, 0.95, coefs_text, transform=ax_stream.transAxes,
                   verticalalignment='top',
                   bbox=box_props,
                   fontsize=30)
    ax_stream.set_aspect('equal')

    plt.ioff()
    plt.show()
