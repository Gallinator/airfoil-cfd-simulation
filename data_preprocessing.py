import multiprocessing
import os.path
import pickle
import random

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


def denormalize_landmarks(landmarks: np.ndarray, scaler) -> np.ndarray:
    feature_mat = landmarks.reshape(-1, 2)
    denorm_feature_mat = scaler.inverse_transform(feature_mat)
    return denorm_feature_mat.reshape(landmarks.shape)


def denormalize_features(u, v, *features, scaler) -> np.ndarray:
    features_shape = np.asarray(features).shape
    feature_mat = np.array(features).reshape(features_shape[0], -1).T
    u_flat, v_flat = np.array(u).flatten(), np.array(v).flatten()
    vel = np.sqrt(np.square(u_flat) + np.square(v_flat)).reshape(-1, 1)

    feature_mat = np.concatenate((feature_mat, vel), axis=1)

    denorm_feature_mat = scaler.inverse_transform(feature_mat)
    u_denorm = (u_flat / scaler.scale_[-1]).reshape(-1, features_shape[2], features_shape[3])
    v_denorm = (v_flat / scaler.scale_[-1]).reshape(-1, features_shape[2], features_shape[3])
    denorm_features = denorm_feature_mat[:, :-1].T.reshape(features_shape)

    return np.append([u_denorm, v_denorm], denorm_features, axis=0)


def denormalize_grid(grid_x: np.ndarray, grid_y: np.ndarray, scaler) -> tuple:
    grid_mat = np.concatenate((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), axis=1)
    denorm_grid_mat = scaler.inverse_transform(grid_mat)
    denorm_grid_x, denorm_grid_y = np.hsplit(denorm_grid_mat, 2)
    return denorm_grid_x.reshape(grid_x.shape), denorm_grid_y.reshape(grid_y.shape)


def get_mask(airfoil_poly: np.ndarray, grid: tuple):
    airfoil_path = matplotlib.path.Path(airfoil_poly)
    grid_x, grid_y = grid
    grid_x, grid_y = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
    grid_points = np.concatenate((grid_x, grid_y), axis=1)
    return airfoil_path.contains_points(grid_points).reshape(grid[0].shape)


def airfoil_sampling_task(i, airfoil_poly, sample_grid, ff):
    i = int(i)
    print(f'Sampling airfoil {i}')
    x = ff['x'][()]
    y = ff['y'][()]
    airfoil_mask = get_mask(airfoil_poly, sample_grid).reshape(sample_grid[0].shape)

    rho_u = sample_gridded_values(sample_grid, ff['rho_u'][()], (x, y))
    rho_u[airfoil_mask] = np.mean(rho_u[~airfoil_mask])

    rho_v = sample_gridded_values(sample_grid, ff['rho_v'][()], (x, y))
    rho_v[airfoil_mask] = np.mean(rho_v[~airfoil_mask])

    rho = sample_gridded_values(sample_grid, ff['rho'][()], (x, y))
    rho[airfoil_mask] = np.mean(rho[~airfoil_mask])

    e = sample_gridded_values(sample_grid, ff['e'][()], (x, y))
    e[airfoil_mask] = np.mean(e[~airfoil_mask])

    return i, rho_u, rho_v, rho, e, 1 * airfoil_mask


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


def normalize_grid(grid_x: np.ndarray, grid_y: np.ndarray, scaler=None) -> tuple:
    grid_mat = np.concatenate((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), axis=1)
    if scaler is None:
        scaler = MinMaxScaler().fit(grid_mat)
    norm_grid_mat = scaler.transform(grid_mat)
    norm_grid_x, norm_grid_y = np.hsplit(norm_grid_mat, 2)
    return norm_grid_x.reshape(grid_x.shape), norm_grid_y.reshape(grid_y.shape), scaler


def normalize_landmarks(landmarks: np.ndarray, grid_scaler):
    feature_mat = landmarks.reshape(-1, 2)
    norm_feature_mat = grid_scaler.transform(feature_mat)
    return norm_feature_mat.reshape(landmarks.shape)


def normalize_features(u: np.ndarray, v: np.ndarray, *features, scaler=None) -> list:
    features_shape = np.asarray(features).shape
    u_flat = np.array(u).flatten()
    v_flat = np.array(v).flatten()
    vel = np.sqrt(np.square(u_flat) + np.square(v_flat)).reshape(-1, 1)
    feature_mat = np.array(features).reshape(features_shape[0], -1).T
    feature_mat = np.concatenate((feature_mat, vel), axis=1)

    feat_scaler = scaler
    if feat_scaler is None:
        feat_scaler = MinMaxScaler().fit(feature_mat)
    norm_feature_mat = feat_scaler.transform(feature_mat)

    u_norm = (u_flat * feat_scaler.scale_[-1]).reshape(-1, features_shape[2], features_shape[3])
    v_norm = (v_flat * feat_scaler.scale_[-1]).reshape(-1, features_shape[2], features_shape[3])
    norm_feature_mat = norm_feature_mat[:, :-1]
    return [u_norm, v_norm] + norm_feature_mat.T.reshape(features_shape).tolist() + [feat_scaler]


def normalize_coefficients(*coefs, scaler=None) -> list:
    num_coefs = len(coefs)
    coefs_mat = np.vstack(coefs).T

    coefs_scaler = scaler
    if coefs_scaler is None:
        coefs_scaler = MinMaxScaler().fit(coefs_mat)

    norm_coefs_mat = coefs_scaler.transform(coefs_mat)
    return np.array(np.vsplit(norm_coefs_mat.T, num_coefs)).squeeze(1).tolist() + [coefs_scaler]


def denormalize_coefficients(*coefs, scaler) -> np.ndarray:
    num_coefs = len(coefs)
    coefs_mat = np.vstack(coefs).T
    norm_coefs_mat = scaler.inverse_transform(coefs_mat)
    return np.array(np.vsplit(norm_coefs_mat.T, num_coefs)).squeeze(1)


def save_scaler(scaler, path: str):
    pickle.dump(scaler, open(path, 'wb'))


def load_scaler(path: str):
    return pickle.load(open(path, 'rb'))


def extract_coefs(src: File, indices, alphas) -> tuple:
    cd = np.array([None for _ in range(len(indices))])
    cl = np.array([None for _ in range(len(indices))])
    cm = np.array([None for _ in range(len(indices))])
    alphas_12 = alphas[indices] == 12
    alphas_04 = alphas[indices] == 4
    cd[alphas_12] = src[f'alpha+12']['C_d'][()][indices][alphas_12]
    cd[alphas_04] = src[f'alpha+04']['C_d'][()][indices][alphas_04]
    cl[alphas_12] = src[f'alpha+12']['C_l'][()][indices][alphas_12]
    cl[alphas_04] = src[f'alpha+04']['C_l'][()][indices][alphas_04]
    cm[alphas_12] = src[f'alpha+12']['C_m'][()][indices][alphas_12]
    cm[alphas_04] = src[f'alpha+04']['C_m'][()][indices][alphas_04]
    return cd, cl, cm


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

        u, v, rho, energy, masks = [[None] * num_samples for _ in range(5)]
        with multiprocessing.Pool() as pool:
            flow_fields = get_flow_fields(source, indices.tolist(), alphas)
            args = [(i, landmarks[i], (grid_x, grid_y), ff[1]) for i, ff in enumerate(flow_fields)]
            for i, r_u, r_v, r, e, m in pool.starmap(airfoil_sampling_task, args):
                u[i] = np.divide(r_u, r, np.zeros_like(r), where=r != 0)
                v[i] = np.divide(r_v, r, np.zeros_like(r), where=r != 0)
                rho[i] = r
                energy[i] = e
                masks[i] = m

        alphas = np.array([int(a) for _, a in enumerate(alphas)])
        c_d, c_l, c_m = extract_coefs(source, indices, alphas)

        train_cd, train_cl, train_cm, coefs_scaler = normalize_coefficients(
            c_d[:train_end], c_l[:train_end], c_m[:train_end])
        norm_grid_x, norm_grid_y, grid_scaler = normalize_grid(grid_x, grid_y)

        norm_landmarks = normalize_landmarks(landmarks, grid_scaler)

        train_u, train_v, train_r, train_e, feature_scaler = normalize_features(
            u[:train_end],
            v[:train_end],
            rho[:train_end],
            energy[:train_end])

    with h5py.File(train_path, 'w') as dest:
        dest['alpha'] = alphas[:train_end]
        dest['landmarks'] = norm_landmarks[:train_end]
        dest['masks'] = masks[:train_end]
        dest['grid'] = np.array([norm_grid_x, norm_grid_y])
        dest['u'] = train_u
        dest['v'] = train_v
        dest['rho'] = train_r
        dest['energy'] = train_e
        dest['C_d'] = train_cd
        dest['C_l'] = train_cl
        dest['C_m'] = train_cm

    test_u, test_v, test_r, test_e, _ = normalize_features(
        u[train_end:],
        v[train_end:],
        rho[train_end:],
        energy[train_end:],
        scaler=feature_scaler)
    test_cd, test_cl, test_cm, _ = normalize_coefficients(c_d[train_end:], c_l[train_end:], c_m[train_end:],
                                                          scaler=coefs_scaler)

    with h5py.File(test_path, 'w') as dest:
        dest['alpha'] = alphas[train_end:]
        dest['masks'] = masks[train_end:]
        dest['landmarks'] = norm_landmarks[train_end:]
        dest['grid'] = np.array([norm_grid_x, norm_grid_y])
        dest['u'] = test_u
        dest['v'] = test_v
        dest['rho'] = test_r
        dest['energy'] = test_e
        dest['C_d'] = test_cd
        dest['C_l'] = test_cl
        dest['C_m'] = test_cm

    save_scaler(grid_scaler, os.path.join(dest_path, 'grid_scaler.pkl'))
    save_scaler(feature_scaler, os.path.join(dest_path, 'features_scaler.pkl'))
    save_scaler(coefs_scaler, os.path.join(dest_path, 'coefs_scaler.pkl'))


def sample_gridded_values(sample_grid: tuple, raw_values, raw_grid: tuple):
    raw_grid_x, raw_grid_y = raw_grid
    sample_grid_x, sample_grid_y = sample_grid
    sampled_values = griddata(np.vstack((raw_grid_x, raw_grid_y.T)).T,
                              raw_values,
                              (sample_grid_x, sample_grid_y),
                              method='nearest')
    return sampled_values


if __name__ == '__main__':
    download_dir = input('Data download directory: ')
    download_data(download_dir)
    data_dir = input('Datasets save directory: ')
    num_samples = int(input('Total samples: '))
    train_size = float(input('Training data proportion [0,1]: '))
    create_sampled_datasets(os.path.join(download_dir, 'airfoil_9k_data.h5'),
                            data_dir,
                            128j,
                            num_samples,
                            train_size)
