import multiprocessing
import os.path

import h5pickle
import h5py
import numpy as np
import requests
import tqdm
from h5py import File
from scipy.interpolate import griddata

from airfoil_dataset import AirfoilDataset
from visualization import plot_airfoil

DATA_URL = 'https://nrel-pds-windai.s3.amazonaws.com/aerodynamic_shapes/2D/9k_airfoils/v1.0.0/airfoil_9k_data.h5'
TRAIN_FILE = 'train_airfoils.h5'
TEST_FILE = 'test_airfoils.h5'


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


def airfoil_sampling_task(i, sample_grid, ff):
    i = int(i)
    print(f'Sampling airfoil {i}')
    x = ff['x'][()]
    y = ff['y'][()]
    r_u = sample_gridded_values(sample_grid, ff['rho_u'][()], (x, y))
    r_v = sample_gridded_values(sample_grid, ff['rho_v'][()], (x, y))
    r = sample_gridded_values(sample_grid, ff['rho'][()], (x, y))
    e = sample_gridded_values(sample_grid, ff['e'][()], (x, y))
    o = sample_gridded_values(sample_grid, ff['omega'][()], (x, y))
    return i, r_u, r_v, r, e, o


def get_flow_fields(src: File, indices):
    ff = []
    for i, item in enumerate(src['alpha+12']['flow_field'].items()):
        if i in indices:
            ff.append(item)
    return ff


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

        rho_u, rho_v, rho, energy, omega = ([None] * num_samples, [None] * num_samples, [None] * num_samples,
                                            [None] * num_samples, [None] * num_samples)
        with multiprocessing.Pool() as pool:
            args = [(i, (grid_x, grid_y), ff[1]) for i, ff in enumerate(get_flow_fields(source, indices))]
            for i, r_u, r_v, r, e, o in pool.starmap(airfoil_sampling_task, args):
                rho_u[i] = r_u
                rho_v[i] = r_v
                rho[i] = r
                energy[i] = e
                omega[i] = o

    with h5py.File(train_path, 'w') as dest:
        dest['landmarks'] = landmarks[:train_end]
        dest['grid'] = np.array([grid_x, grid_y])
        dest['rho_u'] = np.array(rho_u)[:train_end]
        dest['rho_v'] = np.array(rho_v)[:train_end]
        dest['rho'] = np.array(rho)[:train_end]
        dest['energy'] = np.array(energy)[:train_end]
        dest['omega'] = np.array(omega)[:train_end]

    with h5py.File(test_path, 'w') as dest:
        dest['landmarks'] = landmarks[train_end:]
        dest['grid'] = np.array([grid_x, grid_y])
        dest['rho_u'] = np.array(rho_u)[train_end:]
        dest['rho_v'] = np.array(rho_v)[train_end:]
        dest['rho'] = np.array(rho)[train_end:]
        dest['energy'] = np.array(energy)[train_end:]
        dest['omega'] = np.array(omega)[train_end:]


def sample_gridded_values(sample_grid: tuple, raw_values, raw_grid: tuple):
    raw_grid_x, raw_grid_y = raw_grid
    sample_grid_x, sample_grid_y = sample_grid
    sampled_values = griddata(np.vstack((raw_grid_x, raw_grid_y.T)).T,
                              raw_values,
                              (sample_grid_x, sample_grid_y),
                              method='nearest')
    return sampled_values


create_sampled_datasets('/media/luigi/Linux/airfoil_9k_data.h5', 'data', 50j, 1000, 0.8)