import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import torch
from matplotlib.widgets import Slider
from airfoil_dataset import AirfoilDataset
from data_preprocessing import load_scaler, normalize_landmarks, denormalize_features
from model import Model
from airfoil_interactor import AirfoilInteractor

from visualization import plot_airfoil


def get_device():
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Using CPU')
    else:
        print('CUDA is available!  Using GPU')

    return torch.device("cuda:0" if train_on_gpu else "cpu")


device = get_device()

DEFAULT_BEZIER_NODES = [[0.0, 0.0],
                        [0.0, 0.1],
                        [0.25, 0.1],
                        [0.75, 0.05],
                        [1.0, 0.0],
                        [0.75, -0.05],
                        [0.25, -0.1],
                        [0.0, -0.1]]
DEFAULT_ALPHA = 4


def sample_beziers(interactor: AirfoilInteractor) -> np.ndarray:
    samples_u = np.linspace(0.0, 1.0, 501)
    samples_d = np.linspace(0.0, 1.0, 500)
    upper_points, lower_points = interactor.get_bezier_points(samples_u, samples_d)
    return np.concatenate((lower_points.T, upper_points.T))


def deg_to_slope(deg):
    return math.tan(math.radians(deg))


def slope_to_deg(slope):
    return math.degrees(math.atan(slope))


def edit_custom_airfoil() -> tuple:
    upper_polygon = Polygon(np.array(DEFAULT_BEZIER_NODES), animated=True)
    fig, axs = plt.subplots(2, 1, layout='constrained', height_ratios=[0.95, 0.05])
    airfoil_ax, alpha_ax = axs
    alpha_line = airfoil_ax.axline((1.0, 0.0), slope=deg_to_slope(DEFAULT_ALPHA), color='blue')
    airfoil_ax.add_patch(upper_polygon)
    airfoil_interactor = AirfoilInteractor(airfoil_ax, upper_polygon)
    airfoil_ax.set_ylim((-0.5, 0.5))
    airfoil_ax.set_xlim((-0.5, 1.5))
    airfoil_ax.set_aspect('equal')

    alpha_slider = Slider(ax=alpha_ax, label='Angle of attack', valmin=-45, valmax=45, valstep=1, valinit=DEFAULT_ALPHA)

    def update_alpha_line(alpha_deg):
        alpha_line.set_slope(deg_to_slope(alpha_deg))
        fig.canvas.draw_idle()

    alpha_slider.on_changed(update_alpha_line)

    plt.show()
    return sample_beziers(airfoil_interactor), slope_to_deg(alpha_line.get_slope())


def main():
    grid_scaler = load_scaler('data/grid_scaler.pkl')
    features_scaler = load_scaler('data/features_scaler.pkl')
    alpha_scaler = load_scaler('data/alpha_scaler.pkl')

    landmark, alpha = edit_custom_airfoil()

    # Needed only to load the grids
    data = AirfoilDataset('data/test_airfoils.h5')
    grid_x = torch.tensor(data.grid_x, dtype=torch.float32).to(device)
    grid_y = torch.tensor(data.grid_y, dtype=torch.float32).to(device)

    in_size = data.landmarks[0].shape[0] * 2
    model = Model(len(grid_x), in_size)
    model.load_state_dict(torch.load('models/linear.pt'))
    model = model.to(device)

    landmark = torch.tensor(landmark, dtype=torch.float32)
    landmark = landmark.unsqueeze(0).to(device)
    alpha = alpha_scaler.transform([[alpha]])
    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
    g_x = grid_x.unsqueeze(0)
    g_y = grid_y.unsqueeze(0)

    pred_u, pred_v, pred_rho = model.forward(alpha, g_x, g_y, landmark.flatten(start_dim=1))
    pred_u, pred_v, pred_rho = pred_u.numpy(force=True), pred_v.numpy(force=True), pred_rho.numpy(force=True)
    pred_u, pred_v, pred_rho, _, _ = denormalize_features(pred_u,
                                                          pred_v,
                                                          pred_rho,
                                                          np.ones_like(pred_rho),
                                                          np.ones_like(pred_rho),
                                                          scaler=features_scaler)

    plot_airfoil(alpha.numpy(force=True)[0][0],
                 landmark.numpy(force=True)[0],
                 data.grid_x,
                 data.grid_y,
                 pred_u[0], pred_v[0], pred_rho[0])


if __name__ == '__main__':
    main()
