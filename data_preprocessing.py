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


def get_flow_fields(src: File, num_samples: int):
    ff = []
    i = 0
    for item in src['alpha+12']['flow_field'].items():
        if i >= num_samples:
            break
        ff.append(item)
        i += 1
    return ff


def create_sampled_datasets(source_path: str, dest_path: str, sample_grid_size, num_samples: int):
    os.remove(dest_path)
    with (h5pickle.File(source_path, 'r') as source):
        with h5py.File(dest_path, 'w') as dest:
            landmarks = source['shape']['landmarks'][()][:num_samples - 1]
            dest['landmarks'] = landmarks
            grid_x, grid_y = np.mgrid[-0.5:1.5:sample_grid_size, -1:1:sample_grid_size]
            dest['grid'] = np.array([grid_x, grid_y])

            rho_u, rho_v, rho, energy, omega = ([None] * num_samples, [None] * num_samples, [None] * num_samples,
                                                [None] * num_samples, [None] * num_samples)
            with multiprocessing.Pool() as pool:
                args = [(ff[0], (grid_x, grid_y), ff[1]) for ff in get_flow_fields(source, num_samples)]
                for i, r_u, r_v, r, e, o in pool.starmap(airfoil_sampling_task, args):
                    rho_u[i] = r_u
                    rho_v[i] = r_v
                    rho[i] = r
                    energy[i] = e
                    omega[i] = o

                dest['rho_u'] = np.array(rho_u)
                dest['rho_v'] = np.array(rho_v)
                dest['rho'] = np.array(r)
                dest['energy'] = np.array(energy)
                dest['omega'] = np.array(omega)


def sample_gridded_values(sample_grid: tuple, raw_values, raw_grid: tuple):
    raw_grid_x, raw_grid_y = raw_grid
    sample_grid_x, sample_grid_y = sample_grid
    sampled_values = griddata(np.vstack((raw_grid_x, raw_grid_y.T)).T,
                              raw_values,
                              (sample_grid_x, sample_grid_y),
                              method='linear')
    return sampled_values


create_sampled_datasets('/media/luigi/Linux/airfoil_9k_data.h5', 'data/airfoils.h5', 50j, 1000)