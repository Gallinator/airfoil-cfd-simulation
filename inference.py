import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import torch
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
from airfoil_dataset import AirfoilDataset
from data_preprocessing import load_scaler, normalize_landmarks, denormalize_features, get_mask, denormalize_grid
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


def get_axline_transform(ax, alpha):
    return Affine2D().rotate_deg_around(0.5, 0, 90 + alpha) + ax.transData


def edit_custom_airfoil() -> tuple:
    upper_polygon = Polygon(np.array(DEFAULT_BEZIER_NODES), animated=True)
    fig, axs = plt.subplots(2, 1, layout='constrained', height_ratios=[0.95, 0.01])
    airfoil_ax, alpha_ax = axs
    airfoil_ax.add_patch(upper_polygon)
    airfoil_interactor = AirfoilInteractor(airfoil_ax, upper_polygon)
    airfoil_ax.set_ylim((-0.5, 0.5))
    airfoil_ax.set_xlim((-0.5, 1.5))
    alpha_lines = []
    for x in np.linspace(-0.5, 1.5, 20):
        alpha_lines.append(airfoil_ax.axline((x, 0.0), slope=deg_to_slope(DEFAULT_ALPHA), color='paleturquoise',
                                             transform=get_axline_transform(airfoil_ax, DEFAULT_ALPHA)))
    airfoil_ax.set_aspect('equal')

    alpha_slider = Slider(ax=alpha_ax, label='Angle of attack', valmin=-45, valmax=45, valstep=1, valinit=DEFAULT_ALPHA)

    def update_alpha_line(alpha_deg):
        for l in alpha_lines:
            l.set_slope(deg_to_slope(alpha_deg))
            l.set_transform(get_axline_transform(airfoil_ax, alpha_deg))
        fig.canvas.draw_idle()

    alpha_slider.on_changed(update_alpha_line)

    plt.show()
    return sample_beziers(airfoil_interactor), slope_to_deg(alpha_lines[0].get_slope())


def generate_free_flow_grids(alpha, shape):
    grid_x = np.full(shape, math.cos(math.radians(alpha)) * 1)
    grid_x = torch.tensor(grid_x, dtype=torch.float32).unsqueeze(0)
    grid_y = np.full(shape, math.sin(math.radians(alpha)) * 1)
    grid_y = torch.tensor(grid_y, dtype=torch.float32).unsqueeze(0)
    return grid_x, grid_y


def run_inference():
    grid_scaler = load_scaler('data/grid_scaler.pkl')
    features_scaler = load_scaler('data/features_scaler.pkl')

    landmark, alpha = edit_custom_airfoil()

    # Needed only to load the grids
    data = AirfoilDataset('data/test_airfoils.h5')
    norm_landmark = normalize_landmarks(landmark, grid_scaler)
    airfoil_mask = get_mask(norm_landmark, (data.grid_coords_x, data.grid_coords_y))
    airfoil_mask = torch.tensor(airfoil_mask, dtype=torch.float32).to(device).unsqueeze(0)

    model = Model()
    model.load_state_dict(torch.load('models/linear.pt'))
    model = model.to(device)
    model.eval()

    norm_landmark = torch.tensor(norm_landmark, dtype=torch.float32).unsqueeze(0).to(device)
    flow_x, flow_y = generate_free_flow_grids(alpha, data.grid_shape)
    flow_x, flow_y = flow_x.to(device), flow_y.to(device)

    y = model.forward(flow_x, flow_y, norm_landmark.flatten(start_dim=1), airfoil_mask)
    pred_u, pred_v, pred_rho, pred_energy = np.reshape(y.numpy(force=True), ((4, 1) + data.grid_shape))
    pred_u, pred_v, pred_rho, pred_energy = denormalize_features(pred_u, pred_v,
                                                                 pred_rho, pred_energy, scaler=features_scaler)
    grid_coords_x, grid_coords_y = denormalize_grid(data.grid_coords_x, data.grid_coords_y, grid_scaler)

    plot_airfoil(alpha, landmark, airfoil_mask.numpy(force=True)[0],
                 grid_coords_x, grid_coords_y, pred_u[0], pred_v[0], pred_rho[0], pred_energy[0])


if __name__ == '__main__':
    run_inference()
