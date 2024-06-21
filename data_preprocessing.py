import multiprocessing
import os.path

import h5py
import numpy as np
import requests
import tqdm
from scipy.interpolate import griddata

DATA_URL = 'https://nrel-pds-windai.s3.amazonaws.com/aerodynamic_shapes/2D/9k_airfoils/v1.0.0/airfoil_9k_data.h5'


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


def airfoil_sampling_task(i, sample_grid, rho_u, rho_v, r_grid):
    print(f'Sampling {i}')
    rho_u = sample_gridded_values(sample_grid, rho_u, r_grid)
    rho_v = sample_gridded_values(sample_grid, rho_v, r_grid)
    return i, rho_u, rho_v


def create_sampled_datasets(source_path: str, dest_path: str):
    os.remove(dest_path)
    with h5py.File(source_path, 'r') as source:
        with h5py.File(dest_path, 'w') as dest:
            landmarks = source['shape']['landmarks'][()]
            num_airfoils = len(landmarks)
            dest['landmarks'] = landmarks
            grid_x, grid_y = np.mgrid[-0.5:1.5:100j, -1:1:100j]
            dest['grid'] = np.array([grid_x, grid_y])

            raw_grid = []
            raw_rho_u = []
            raw_rho_v = []
            for i, ff in tqdm.tqdm(source['alpha+04']['flow_field'].items(), desc='Reading flow fields into memory'):
                raw_grid.append((ff['x'][()], ff['y'][()]))
                raw_rho_u.append(ff['rho_u'][()])
                raw_rho_v.append(ff['rho_v'][()])

            momentum_x, momentum_y = [None] * num_airfoils, [None] * num_airfoils
            with multiprocessing.Pool() as pool:
                args = [(i, (grid_x, grid_y), raw_rho_u[i], raw_rho_v[i], raw_grid[i]) for i in range(num_airfoils)]
                for i, m_x, m_y in pool.starmap(airfoil_sampling_task, args):
                    momentum_x[i] = m_x
                    momentum_y[i] = m_y

                dest['rho_u'] = np.array(momentum_x)
                dest['rho_v'] = np.array(momentum_y)


def sample_gridded_values(sample_grid: tuple, raw_values, raw_grid: tuple):
    raw_grid_x, raw_grid_y = raw_grid
    sample_grid_x, sample_grid_y = sample_grid
    sampled_values = griddata(np.vstack((raw_grid_x, raw_grid_y.T)).T,
                              raw_values,
                              (sample_grid_x, sample_grid_y),
                              method='linear')
    return sampled_values
