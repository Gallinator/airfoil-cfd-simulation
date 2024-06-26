import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

COLORMAP = 'jet'
plt.style.use('dark_background')


def plot_training_history(losses, lr):
    x = np.arange(len(losses))
    fig, axs = plt.subplots(2)

    axs[0].set_title('Training loss')
    axs[0].set_xlabel('Epoch')
    axs[0].plot(x, losses)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].set_title('Training learning rate')
    axs[1].set_xlabel('Epoch')
    axs[1].plot(x, lr)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()


def plot_airfoil(landmarks: np.ndarray, grid_x: np.array, grid_y: np.array, momentum_x: np.array, momentum_y: np.array):
    # Use the momentum magnitude as color
    color = np.sqrt(np.square(momentum_y) + np.square(momentum_x))
    vmin, vmax = np.min(color), np.max(color)

    fig, ax = plt.subplots(1)
    ax.set_title("Flow momentum")
    ax.fill(landmarks[:, 0], landmarks[:, 1], color='grey')
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
