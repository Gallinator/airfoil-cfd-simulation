import argparse
import math
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import torch
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
from data_preprocessing import load_scaler, normalize_landmarks, denormalize_features, get_mask, denormalize_grid, \
    denormalize_coefficients
from model import Model
from airfoil_interactor import AirfoilInteractor
from utils import get_torch_device

from visualization import plot_airfoil

device = get_torch_device()

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
    """
    :param interactor: Airfoil editor
    :return: a (1001,2) array containing the sampled points from the airfoil editor current curve
    """
    samples_u = np.linspace(0.0, 1.0, 501)
    samples_d = np.linspace(0.0, 1.0, 500)
    upper_points, lower_points = interactor.get_bezier_points(samples_u, samples_d)
    return np.concatenate((lower_points.T, upper_points.T))


def deg_to_slope(deg):
    return math.tan(math.radians(deg))


def slope_to_deg(slope):
    return math.degrees(math.atan(slope))


def get_axline_transform(ax, alpha):
    """
    Creates a transform to keep the airflow lines equally spaced.
    :param ax: axis
    :param alpha: angle of attack
    :return: the transform
    """
    return Affine2D().rotate_deg_around(0.5, 0, 90 + alpha) + ax.transData


def edit_custom_airfoil() -> tuple:
    """
    Shows the airfoil editor.
    :return: a (1001,2) aray containing the airfoil landmarks and the angle of attack in degrees
    """
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

    alpha_slider = Slider(ax=alpha_ax, label='Angle of attack', valmin=-4, valmax=20, valstep=1, valinit=DEFAULT_ALPHA)

    def update_alpha_line(alpha_deg):
        for l in alpha_lines:
            l.set_slope(deg_to_slope(alpha_deg))
            l.set_transform(get_axline_transform(airfoil_ax, alpha_deg))
        fig.canvas.draw_idle()

    alpha_slider.on_changed(update_alpha_line)

    plt.show()
    return sample_beziers(airfoil_interactor), slope_to_deg(alpha_lines[0].get_slope())


def generate_free_flow_grids(alpha, shape):
    """

    :param alpha: angle of attack
    :param shape: the shape of the grid, usually (128,128)
    :return: two (N,N) array containing x and y free flow velocity grids
    """
    grid_x = np.full(shape, math.cos(math.radians(alpha)) * 1)
    grid_x = torch.tensor(grid_x, dtype=torch.float32).unsqueeze(0)
    grid_y = np.full(shape, math.sin(math.radians(alpha)) * 1)
    grid_y = torch.tensor(grid_y, dtype=torch.float32).unsqueeze(0)
    return grid_x, grid_y


def run_inference(data_path: str, model_path: str):
    grid_scaler = load_scaler(os.path.join(data_path, 'grid_scaler.pkl'))
    features_scaler = load_scaler(os.path.join(data_path, 'features_scaler.pkl'))
    coefs_scaler = load_scaler(os.path.join(data_path, 'coefs_scaler.pkl'))

    landmark, alpha = edit_custom_airfoil()

    grid_coords_x, grid_coords_y = np.load(os.path.join(data_path, 'grid_coords.npy'))
    grid_shape = grid_coords_x.shape
    norm_landmark = normalize_landmarks(landmark.copy(), grid_scaler)
    airfoil_mask = get_mask(norm_landmark, (grid_coords_x, grid_coords_y))
    airfoil_mask = torch.tensor(airfoil_mask, dtype=torch.float32).unsqueeze(0)

    model = Model()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    flow_x, flow_y = generate_free_flow_grids(alpha, grid_shape)
    flow_data = torch.stack((flow_x, flow_y, airfoil_mask), 1).to(device)

    pred_flow, pred_coefs = model.forward(flow_data)
    pred_u, pred_v, pred_rho, pred_energy = np.reshape(pred_flow.numpy(force=True), ((4, 1) + grid_shape))
    pred_u, pred_v, pred_rho, pred_energy = denormalize_features(pred_u, pred_v,
                                                                 pred_rho, pred_energy, scaler=features_scaler)
    grid_coords_x, grid_coords_y = denormalize_grid(grid_coords_x, grid_coords_y, grid_scaler)

    cd, cl, cm = pred_coefs.flatten().numpy(force=True)
    cd, cl, cm = denormalize_coefficients(cd, cl, cm, scaler=coefs_scaler)

    plot_airfoil(alpha, landmark, airfoil_mask.numpy(force=True)[0],
                 grid_coords_x, grid_coords_y, pred_u[0], pred_v[0], pred_rho[0], pred_energy[0],
                 cl[0], cd[0], cm[0])


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', '-d', type=str, default='data',
                            help='directory containing the feature scalers and grid coordinates')
    arg_parser.add_argument('--weights-dir', '-w', type=str, default='models',
                            help='directory containing the model weights')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    run_inference(args.data_dir, args.weights_dir)
