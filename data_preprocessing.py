import argparse
import multiprocessing
import os.path
import pickle
import random

import h5pickle
import h5py
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import requests
import tqdm
from h5py import File
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler

from visualization import plot_raw_data, plot_airfoil

DATA_URL = 'https://nrel-pds-windai.s3.amazonaws.com/aerodynamic_shapes/2D/9k_airfoils/v1.0.0/airfoil_9k_data.h5'
TRAIN_FILE = 'train_airfoils.h5'
EVAL_FILE = 'eval_airfoils.h5'
AIRFOIL_MASK_VALUE = 0


def download_data(dest_dir: str):
    """
    Downloads the data to a file named airfoil_9k_data.h5
    :param dest_dir: the download directory path
    """
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


def denormalize_features(u: np.ndarray, v: np.ndarray, *features, scaler) -> np.ndarray:
    """
    :param u: an (N,128,128) array containing the x velocities
    :param v: an (N,128,128) array containing the x velocities
    :param features: other features passed as (N,128,128) arrays
    :param scaler: the training features scaler
    :return: an array containing the denormalized features in the same order they were passed
    """
    features_shape = np.shape(features)
    feature_mat = np.asarray(features).reshape(features_shape[0], -1).T
    u_flat, v_flat = u.ravel(), v.ravel()
    vel = np.sqrt(np.square(u_flat) + np.square(v_flat)).reshape(-1, 1)

    feature_mat = np.concatenate((feature_mat, vel), axis=1)

    denorm_feature_mat = scaler.inverse_transform(feature_mat)
    u_denorm = (u_flat / scaler.scale_[-1]).reshape(-1, features_shape[2], features_shape[3])
    v_denorm = (v_flat / scaler.scale_[-1]).reshape(-1, features_shape[2], features_shape[3])
    denorm_features = denorm_feature_mat[:, :-1].T.reshape(features_shape)

    return np.append([u_denorm, v_denorm], denorm_features, axis=0)


def denormalize_grid(grid_x: np.ndarray, grid_y: np.ndarray, scaler) -> tuple:
    """
    :param grid_x: (128,128) array containing the x coordinates of grid points
    :param grid_y: (128,128) array containing the y coordinates of grid points
    :param scaler: the training grid scaler
    :return: denormalized x and y coordinates grids
    """
    grid_mat = np.concatenate((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), axis=1)
    denorm_grid_mat = scaler.inverse_transform(grid_mat)
    denorm_grid_x, denorm_grid_y = np.hsplit(denorm_grid_mat, 2)
    return denorm_grid_x.reshape(grid_x.shape), denorm_grid_y.reshape(grid_y.shape)


def get_mask(airfoil_poly: np.ndarray, grid: tuple):
    """
    Generates a binary mask for the airfoil. The mask has value 1 were the airfoil is, 0 elsewhere
    :param airfoil_poly: an (N, 2) array of airfoil points coordinates
    :param grid: a tuple containing the x and y grid coordinates
    :return: the airfoil mask
    """
    airfoil_path = matplotlib.path.Path(airfoil_poly)
    grid_x, grid_y = grid
    grid_x, grid_y = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
    grid_points = np.concatenate((grid_x, grid_y), axis=1)
    return airfoil_path.contains_points(grid_points).reshape(grid[0].shape)


def airfoil_sampling_task(i, airfoil_poly, sample_grid, ff):
    """
    Function to be executed on the multiprocessing pool.
    Extracts a sample from the h5 file.
    :param i: the sample index. This is returned to reconstruct the samples order when collecting the results.
    :param airfoil_poly: an (N, 2) array of airfoil points coordinates
    :param sample_grid: a tuple containing the x and y grid coordinates
    :param ff: the 'fuel_flow' group from the h5 file
    :return: a tuple containing the index, density*u, density*v, density, energy and mask
    """
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


def get_flow_fields(src: File, indices) -> list:
    """
    Extracts the 'fuel_flow' groups from the h5 file.
    As h5py groups do not support slicing the only way to select item is to iterate over the whole group
    :param src: The angle of attack group
    :param indices: The indices to extract the group of
    :return: a list of 'fuel_flow' groups
    """

    ff = [None for _ in range(len(indices))]
    for i, item in enumerate(src['flow_field'].items()):
        try:
            idx = indices.index(i)
        except ValueError:
            continue
        ff[idx] = item
    return ff


def normalize_grid(grid_x: np.ndarray, grid_y: np.ndarray, scaler=None) -> tuple:
    """
    Normalizes the grid coordinate with a MinMaxScaler
    :param grid_x: raw grid x coordinates
    :param grid_y: raw grid x coordinates
    :param scaler: the training scaler or None to create a new scaler
    :return: the normalized grid coordinates
    """
    grid_mat = np.concatenate((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), axis=1)
    if scaler is None:
        scaler = MinMaxScaler(copy=False).fit(grid_mat)
    norm_grid_mat = scaler.transform(grid_mat)
    norm_grid_x, norm_grid_y = np.hsplit(norm_grid_mat, 2)
    return norm_grid_x.reshape(grid_x.shape), norm_grid_y.reshape(grid_y.shape), scaler


def normalize_landmarks(landmarks: np.ndarray, grid_scaler):
    """
    :param landmarks: an (N,2) array containing the airfoil vertices coordinates
    :param grid_scaler: the grid scaler
    :return: the normalize airfoil landmarks
    """
    feature_mat = landmarks.reshape(-1, 2)
    norm_feature_mat = grid_scaler.transform(feature_mat)
    return norm_feature_mat.reshape(landmarks.shape)


def normalize_features(u: np.ndarray, v: np.ndarray, *features, scaler=None) -> list:
    """
    Normalizes the features with a MinMaxScaler.
    The velocity x and y components are scaled according to their normalized magnitude to keep directional information.
    :param u: an (N,128,128) array containing the x velocities
    :param v: an (N,128,128) array containing the y velocities
    :param features: other features passed as (N,128,128) arrays
    :param scaler: the training features scaler or None to create a new scaler
    :return: an array containing the normalized features in the same order they were passed and the scaler
    """
    features_shape = np.shape(features)
    u_flat = u.ravel()
    v_flat = v.ravel()
    vel = np.sqrt(np.square(u_flat) + np.square(v_flat)).reshape(-1, 1)
    feature_mat = np.asarray(features).reshape(features_shape[0], -1).T
    feature_mat = np.concatenate((feature_mat, vel), axis=1)

    feat_scaler = scaler
    if feat_scaler is None:
        feat_scaler = MinMaxScaler(copy=False).fit(feature_mat)
    norm_feature_mat = feat_scaler.transform(feature_mat)

    u_norm = (u_flat * feat_scaler.scale_[-1] + feat_scaler.min_[-1]).reshape(-1, features_shape[2], features_shape[3])
    v_norm = (v_flat * feat_scaler.scale_[-1] + feat_scaler.min_[-1]).reshape(-1, features_shape[2], features_shape[3])
    norm_feature_mat = norm_feature_mat[:, :-1]
    return [u_norm, v_norm] + [*norm_feature_mat.T.reshape(features_shape)] + [feat_scaler]


def normalize_coefficients(*coefs, scaler=None) -> list:
    """
    Normalizes the lift, drag and momentum coefficients with a MinMaxScaler
    :param coefs: coefficients passed as arrays
    :param scaler: the training scaler or None to create a new scaler
    :return: a list containing the normalized coefficients in the order they were passed and the scaler
    """
    num_coefs = len(coefs)
    coefs_mat = np.vstack(coefs).T

    coefs_scaler = scaler
    if coefs_scaler is None:
        coefs_scaler = MinMaxScaler(copy=False).fit(coefs_mat)

    norm_coefs_mat = coefs_scaler.transform(coefs_mat)
    return [*np.asarray(np.vsplit(norm_coefs_mat.T, num_coefs)).squeeze(1)] + [coefs_scaler]


def denormalize_coefficients(*coefs, scaler) -> np.ndarray:
    """
    :param coefs: normalized coefficients passed as arrays
    :param scaler: the training scaler
    :return: a list containing the normalized coefficients in the order they were passed
    """
    num_coefs = len(coefs)
    coefs_mat = np.vstack(coefs).T
    norm_coefs_mat = scaler.inverse_transform(coefs_mat)
    return np.asarray(np.vsplit(norm_coefs_mat.T, num_coefs)).squeeze(1)


def save_scaler(scaler, path: str):
    pickle.dump(scaler, open(path, 'wb'))


def load_scaler(path: str):
    return pickle.load(open(path, 'rb'))


def extract_coefs(src: File, indices) -> tuple:
    """
    Extract coefficients from an angle of attack group
    :param src: the angle off attack group
    :param indices: the samples to extract
    :return: the extracted drag, lift and momentum coefficients
    """
    cd = src['C_d'][()][indices]
    cl = src['C_l'][()][indices]
    cm = src['C_m'][()][indices]
    return cd, cl, cm


def shuffle_data(*data: np.ndarray):
    """
    Randomly shuffles data in place
    :param data: the data to shuffle
    :return: the shuffled data
    """
    seed = np.random.randint(0, 2 ** (32 - 1) - 1)
    for d in data:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(d)


def generate_indices(data_len, num_samples):
    """
    Generates unique random indices in [0,data_len)
    :param data_len: the size of the data
    :param num_samples: the number of samples to generate
    :return: the generated indices
    """
    indices = np.arange(data_len, dtype=int)
    np.random.shuffle(indices)
    return indices[:num_samples]


def create_sampled_datasets(source_path: str, dest_path: str, sample_grid_size, num_samples: int, train_size: float):
    """
    Creates the training and evaluation datasets. Also creates the feature scalers and a numpy file containing the normalized x and y grid coordinates.
    The samples are drawn equally from the 4 and 12 angle of attack datasets.
    The processing is done in parallel to improve performance and array copying is avoided as much as possible.
    Still a lot of memory is required to generate large datasets.
    :param source_path: the directory containing the raw data
    :param dest_path: the directory to save the generated files
    :param sample_grid_size: the size of the sampling grid. The model accepts (128,128) grids
    :param num_samples: the total number of samples of the generated data
    :param train_size: the proportion of data to use for the training set in [0,1]
    """
    train_path = os.path.join(dest_path, TRAIN_FILE)
    eval_path = os.path.join(dest_path, EVAL_FILE)
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(eval_path):
        os.remove(eval_path)

    with h5pickle.File(source_path, 'r') as source:
        landmarks = source['shape']['landmarks'][()]
        data_len = len(landmarks)
        per_alpha_samples = num_samples // 2
        indices_04 = generate_indices(data_len, per_alpha_samples)
        indices_12 = generate_indices(data_len, per_alpha_samples)
        all_indices = np.concatenate([indices_04, indices_12])
        landmarks = landmarks[all_indices]
        alphas = ['04'] * per_alpha_samples + ['12'] * per_alpha_samples
        train_end = int(num_samples * train_size)

        grid_x, grid_y = np.mgrid[-0.5:1.5:sample_grid_size, -1:1:sample_grid_size]

        u, v, rho, energy, masks = [[None] * num_samples for _ in range(5)]
        with multiprocessing.Pool() as pool:
            flow_fields = (get_flow_fields(source['alpha+04'], indices_04.tolist()) +
                           get_flow_fields(source['alpha+12'], indices_12.tolist()))
            args = [(i, landmarks[i], (grid_x, grid_y), ff[1]) for i, ff in enumerate(flow_fields)]
            for i, r_u, r_v, r, e, m in pool.starmap(airfoil_sampling_task, args):
                u[i] = np.divide(r_u, r, np.zeros_like(r), where=r != 0)
                v[i] = np.divide(r_v, r, np.zeros_like(r), where=r != 0)
                rho[i] = r
                energy[i] = e
                masks[i] = m

        u = np.asarray(u)
        v = np.asarray(v)
        rho = np.asarray(rho)
        energy = np.asarray(energy)

        alphas = np.asarray([int(a) for _, a in enumerate(alphas)])
        coefs_04 = extract_coefs(source['alpha+04'], indices_04)
        coefs_12 = extract_coefs(source['alpha+12'], indices_12)
        c_d, c_l, c_m = np.concatenate((coefs_04, coefs_12), 1)

        shuffle_data(landmarks, alphas, masks, u, v, rho, energy, c_d, c_l, c_m)

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
        dest['grid'] = np.asarray([norm_grid_x, norm_grid_y])
        dest['u'] = train_u
        dest['v'] = train_v
        dest['rho'] = train_r
        dest['energy'] = train_e
        dest['C_d'] = train_cd
        dest['C_l'] = train_cl
        dest['C_m'] = train_cm

    eval_u, eval_v, eval_r, eval_e, _ = normalize_features(
        u[train_end:],
        v[train_end:],
        rho[train_end:],
        energy[train_end:],
        scaler=feature_scaler)
    eval_cd, eval_cl, eval_cm, _ = normalize_coefficients(c_d[train_end:], c_l[train_end:], c_m[train_end:],
                                                          scaler=coefs_scaler)

    with h5py.File(eval_path, 'w') as dest:
        dest['alpha'] = alphas[train_end:]
        dest['masks'] = masks[train_end:]
        dest['landmarks'] = norm_landmarks[train_end:]
        dest['grid'] = np.asarray([norm_grid_x, norm_grid_y])
        dest['u'] = eval_u
        dest['v'] = eval_v
        dest['rho'] = eval_r
        dest['energy'] = eval_e
        dest['C_d'] = eval_cd
        dest['C_l'] = eval_cl
        dest['C_m'] = eval_cm

    save_scaler(grid_scaler, os.path.join(dest_path, 'grid_scaler.pkl'))
    save_scaler(feature_scaler, os.path.join(dest_path, 'features_scaler.pkl'))
    save_scaler(coefs_scaler, os.path.join(dest_path, 'coefs_scaler.pkl'))
    np.save(os.path.join(dest_path, 'grid_coords.npy'), np.asarray([norm_grid_x, norm_grid_y]))


def sample_gridded_values(sample_grid: tuple, raw_values, raw_grid: tuple):
    """
    Creates an equally spaced grid from a raw grid. Sampling is done using the nearest point.
    :param sample_grid: the x and y coordinates of the sampling grid
    :param raw_values: the raw values of the feature to sample
    :param raw_grid: the x and y coordinates of the raw grid
    :return: a grid of sampled values
    """
    raw_grid_x, raw_grid_y = raw_grid
    sample_grid_x, sample_grid_y = sample_grid
    sampled_values = griddata(np.vstack((raw_grid_x, raw_grid_y.T)).T,
                              raw_values,
                              (sample_grid_x, sample_grid_y),
                              method='nearest')
    return sampled_values


def show_raw_data_example(file_path: str):
    with h5py.File(file_path, 'r') as src:
        i = random.randint(0, len(src['shape']['landmarks']))
        landmark = src['shape']['landmarks'][i][()]
        alpha = random.choice(['04', '12'])
        ff = src[f'alpha+{alpha}']['flow_field'][str(i).rjust(4, '0')]
        grid_x, grid_y = ff['x'][()], ff['y'][()]
        limits = (grid_x <= 1.5) & (grid_x >= -0.5) & (grid_y >= -1) & (grid_y <= 1)
        grid_x, grid_y = grid_x[limits], grid_y[limits]
        rho_u = ff['rho_u'][()][limits]
        rho_v = ff['rho_v'][()][limits]
        rho = ff['rho'][()][limits]
        energy = ff['e'][()][limits]
        omega = ff['omega'][()][limits]
        cl = src[f'alpha+{alpha}']['C_l'][i]
        cd = src[f'alpha+{alpha}']['C_d'][i]
        cm = src[f'alpha+{alpha}']['C_m'][i]
    plot_raw_data(alpha, grid_x, grid_y, landmark, rho_u, rho_v, rho, energy, omega, cl, cd, cm)
    plt.show()


def show_normalized_data_sample(file_path: str):
    with h5py.File(file_path, 'r') as src:
        i = random.randint(0, len(src['alpha']) - 1)
        alpha = src['alpha'][i]
        landmark = src['landmarks'][i][()]
        grid_x, grid_y = src['grid'][()]
        u = src['u'][i][()]
        v = src['v'][i][()]
        rho = src['rho'][i][()]
        energy = src['energy'][i][()]
        cl = src['C_l'][i]
        cd = src['C_d'][i]
        cm = src['C_m'][i]
        mask = src['masks']
    plot_airfoil(alpha, landmark, mask, grid_x, grid_y, u, v, rho, energy, cl, cd, cm)
    plt.show()


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', '-d', type=str, default='data',
                            help='directory to store the preprocessed data into')
    arg_parser.add_argument('--download-dir', '-dl', type=str, default='data',
                            help='directory to store the downloaded data into. The download size is 52.7 Gb')
    arg_parser.add_argument('--num-samples', '-ns', type=int, default=8996,
                            help='total size of the preprocessed data')
    arg_parser.add_argument('--train-size', '-ts', type=float, default=0.8,
                            help='size of the train set. Must be in [0,1]')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    download_dir = args.download_dir
    data_dir = args.data_dir

    download_data(download_dir)
    show_raw_data_example(os.path.join(download_dir, 'airfoil_9k_data.h5'))
    create_sampled_datasets(os.path.join(download_dir, 'airfoil_9k_data.h5'),
                            data_dir,
                            128j,
                            args.num_samples,
                            args.train_size)
    show_normalized_data_sample(os.path.join(data_dir, EVAL_FILE))
