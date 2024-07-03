import multiprocessing
import os.path
import random

import h5pickle
import h5py
import matplotlib.path
import numpy as np
import requests
import tqdm
from h5py import File
from scipy.interpolate import griddata

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


def get_flow_fields(src: File, indices, alphas) -> list:
    # As h5py groups do not support slicing the only way to select item is to iterate over the whole group
    ff = [None for _ in range(len(indices))]
    for i, item in enumerate(src[f'alpha+12']['flow_field'].items()):
        try:
            idx = indices.index(i)
        except ValueError:
            continue
        if alphas[idx] == '12':
            ff[idx] = item
    for i, item in enumerate(src[f'alpha+04']['flow_field'].items()):
        try:
            idx = indices.index(i)
        except ValueError:
            continue
        if alphas[idx] == '04':
            ff[idx] = item
    return ff


def create_sampled_datasets(source_path: str, dest_path: str, sample_grid_size, num_samples: int, train_size: float):
    train_path = os.path.join(dest_path, TRAIN_FILE)
    test_path = os.path.join(dest_path, TEST_FILE)
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(test_path):
        os.remove(test_path)

    with (h5pickle.File(source_path, 'r') as source):
        landmarks = source['shape']['landmarks'][()]
        num_airfoils = len(landmarks)

        indices = np.arange(num_airfoils, dtype=int)
        np.random.shuffle(indices)
        indices = indices[:num_samples]
        landmarks = landmarks[indices]
        alphas = random.choices(['04', '12'], k=num_airfoils)
        train_end = int(num_samples * train_size)

        grid_x, grid_y = np.mgrid[-0.5:1.5:sample_grid_size, -1:1:sample_grid_size]

        u, v, p, energy, omega = [[None] * num_samples for _ in range(5)]
        with multiprocessing.Pool() as pool:
            flow_fields = get_flow_fields(source, indices.tolist(), alphas)
            args = [(i, landmarks[i], (grid_x, grid_y), ff[1]) for i, ff in enumerate(flow_fields)]
            for i, r_u, r_v, r, e, o in pool.starmap(airfoil_sampling_task, args):
                r_raw = r.flatten()
                u_raw = np.divide(r_u.flatten(), r_raw, np.zeros_like(r_raw), where=r_raw != 0)
                v_raw = np.divide(r_v.flatten(), r_raw, np.zeros_like(r_raw), where=r_raw != 0)
                u[i] = u_raw
                v[i] = v_raw
                p[i] = 1 / 2 * r_raw * (u_raw ** 2 + v_raw ** 2)
                energy[i] = e.flatten()
                omega[i] = o.flatten()

        alphas = [int(a) for _, a in enumerate(alphas)]

    with h5py.File(train_path, 'w') as dest:
        dest['alpha'] = alphas[:train_end]
        dest['landmarks'] = landmarks[:train_end]
        dest['grid'] = np.array([grid_x.flatten(), grid_y.flatten()])
        dest['u'] = u[:train_end]
        dest['v'] = v[:train_end]
        dest['p'] = p[:train_end]
        dest['energy'] = energy[:train_end]
        dest['omega'] = omega[:train_end]

    with h5py.File(test_path, 'w') as dest:
        dest['alpha'] = alphas[train_end:]
        dest['landmarks'] = landmarks[train_end:]
        dest['grid'] = np.array([grid_x.flatten(), grid_y.flatten()])
        dest['u'] = u[train_end:]
        dest['v'] = v[train_end:]
        dest['p'] = p[train_end:]
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