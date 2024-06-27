import multiprocessing
import os.path
import sys

import h5pickle
import h5py
import matplotlib.path
import numpy as np
import requests
import tqdm
from h5py import File
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler

from airfoil_dataset import AirfoilDataset
from visualization import plot_airfoil

DATA_URL = 'https://nrel-pds-windai.s3.amazonaws.com/aerodynamic_shapes/2D/9k_airfoils/v1.0.0/airfoil_9k_data.h5'
TRAIN_FILE = 'train_airfoils.h5'
TEST_FILE = 'test_airfoils.h5'
AIRFOIL_MASK_VALUE = 0


def download_data(dest_dir: str):
    url_resource_name = os.path.basename(DATA_URL)
    file_path = os.path.join(dest_dir, url_resource_name)
    if os.path.exists(file_path):
        print(f'The file {file_path} already exists, skipping download')
        return
    with open(file_path, 'wb') as file:
        with requests.get(DATA_URL, stream=True) as res:
            res.raise_for_status()
            total_b = int(res.headers.get('content-length', 0))
            with tqdm.tqdm(desc=url_resource_name, total=total_b, unit='b', unit_divisor=1024,
                           unit_scale=1) as progress:
                for chunk in res.iter_content(chunk_size=8192):
                    progress.update(len(chunk))
                    file.write(chunk)


def get_mask(airfoil_poly: np.ndarray, grid: tuple):
    airfoil_path = matplotlib.path.Path(airfoil_poly)
    grid_x, grid_y = grid
    grid_x, grid_y = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
    grid_points = np.concatenate((grid_x, grid_y), axis=1)
    return airfoil_path.contains_points(grid_points)


def airfoil_sampling_task(i, airfoil_poly, sample_grid, ff):
    i = int(i)
    print(f'Sampling airfoil {i}')
    x = ff['x'][()]
    y = ff['y'][()]
    airfoil_mask = get_mask(airfoil_poly, sample_grid).reshape(sample_grid[0].shape)

    r_u = sample_gridded_values(sample_grid, ff['rho_u'][()], (x, y))
    r_u[airfoil_mask] = AIRFOIL_MASK_VALUE

    r_v = sample_gridded_values(sample_grid, ff['rho_v'][()], (x, y))
    r_v[airfoil_mask] = AIRFOIL_MASK_VALUE

    r = sample_gridded_values(sample_grid, ff['rho'][()], (x, y))
    r[airfoil_mask] = AIRFOIL_MASK_VALUE

    e = sample_gridded_values(sample_grid, ff['e'][()], (x, y))
    e[airfoil_mask] = AIRFOIL_MASK_VALUE

    o = sample_gridded_values(sample_grid, ff['omega'][()], (x, y))
    o[airfoil_mask] = AIRFOIL_MASK_VALUE

    return i, r_u, r_v, r, e, o


def get_flow_fields(src: File, indices):
    ff = []
    for i, item in enumerate(src['alpha+12']['flow_field'].items()):
        if i in indices:
            ff.append(item)
    return ff


def normalize_grid(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple:
    grid_mat = np.concatenate((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), axis=1)
    scaler = MinMaxScaler().fit(grid_mat)
    norm_grid_mat = scaler.transform(grid_mat)
    grid_x, grid_y = np.hsplit(norm_grid_mat, 2)
    return grid_x.flatten(), grid_y.flatten(), scaler


def normalize_landmarks(landmarks: np.ndarray, grid_scaler):
    feature_mat = landmarks.reshape(-1, 2)
    norm_feature_mat = grid_scaler.transform(feature_mat)
    return norm_feature_mat.reshape(landmarks.shape)


def create_sampled_datasets(source_path: str, dest_path: str, sample_grid_size, num_samples: int, train_size: float):
    train_path = os.path.join(dest_path, TRAIN_FILE)
    test_path = os.path.join(dest_path, TEST_FILE)
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(test_path):
        os.remove(test_path)

    with (h5pickle.File(source_path, 'r') as source):
        landmarks = source['shape']['landmarks'][()][:num_samples]
        num_airfoils = len(landmarks)

        indices = np.arange(num_airfoils, dtype=int)
        np.random.shuffle(indices)
        indices = indices[:num_samples]
        landmarks = landmarks[indices]
        train_end = int(num_samples * train_size)

        grid_x, grid_y = np.mgrid[-0.5:1.5:sample_grid_size, -1:1:sample_grid_size]

        rho_u, rho_v, rho, energy, omega = [[None] * num_samples for _ in range(5)]
        with multiprocessing.Pool() as pool:
            args = [(i, landmarks[i], (grid_x, grid_y), ff[1]) for i, ff in enumerate(get_flow_fields(source, indices))]
            for i, r_u, r_v, r, e, o in pool.starmap(airfoil_sampling_task, args):
                rho_u[i] = r_u.flatten()
                rho_v[i] = r_v.flatten()
                rho[i] = r.flatten()
                energy[i] = e.flatten()
                omega[i] = o.flatten()

    grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
    norm_grid_x, norm_grid_y, grid_scaler = normalize_grid(grid_x, grid_y)
    norm_landmarks = normalize_landmarks(landmarks, grid_scaler)

    with h5py.File(train_path, 'w') as dest:
        dest['landmarks'] = norm_landmarks[:train_end]
        dest['grid'] = np.array([norm_grid_x, norm_grid_y])
        dest['rho_u'] = rho_u[:train_end]
        dest['rho_v'] = rho_v[:train_end]
        dest['rho'] = rho[:train_end]
        dest['energy'] = energy[:train_end]
        dest['omega'] = omega[:train_end]

    with h5py.File(test_path, 'w') as dest:
        dest['landmarks'] = norm_landmarks[train_end:]
        dest['grid'] = np.array([norm_grid_x, norm_grid_y])
        dest['rho_u'] = rho_u[train_end:]
        dest['rho_v'] = rho_v[train_end:]
        dest['rho'] = rho[train_end:]
        dest['energy'] = energy[train_end:]
        dest['omega'] = omega[train_end:]


def sample_gridded_values(sample_grid: tuple, raw_values, raw_grid: tuple):
    raw_grid_x, raw_grid_y = raw_grid
    sample_grid_x, sample_grid_y = sample_grid
    sampled_values = griddata(np.vstack((raw_grid_x, raw_grid_y.T)).T,
                              raw_values,
                              (sample_grid_x, sample_grid_y),
                              method='nearest')
    return sampled_values


create_sampled_datasets('/media/luigi/Linux/airfoil_9k_data.h5', 'data', 50j, 1000, 0.8)