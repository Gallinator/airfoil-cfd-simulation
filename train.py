import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from airfoil_dataset import AirfoilDataset
from loss_tracker import LossTracker
from model import Model
from utils import get_torch_device
from visualization import plot_training_history

device = get_torch_device()


def train_model(save_path: str, data_path: str):
    train_data = AirfoilDataset(os.path.join(data_path, 'train_airfoils.h5'))
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    model = model.to(device)
    model.train()

    epochs = 120
    optimizer = AdamW(model.parameters(), lr=0.0001)
    loss = MSELoss()

    loss_tracker = LossTracker('Total')

    for e in range(epochs):

        prog = tqdm.tqdm(train_loader, desc=f'Epoch {e}')
        for batch in prog:
            optimizer.zero_grad()

            grid_x, grid_y, landmarks, u, v, rho, energy, mask, cd, cl, cm = batch
            landmarks = landmarks.flatten(start_dim=1).to(device)
            grid_x = grid_x.to(device)
            grid_y = grid_y.to(device)
            u = u.to(device)
            v = v.to(device)
            rho = rho.to(device)
            energy = energy.to(device)
            mask = mask.to(device)
            coefs = torch.cat((cl, cd, cm), 1).to(device)
            label = torch.stack((u, v, rho, energy), 1)

            y = model.forward(grid_x, grid_y, landmarks, mask)

            batch_loss = loss(label, y)
            batch_loss.backward()
            optimizer.step()

            loss_tracker.batch_update(Total=batch_loss.item())

        loss_tracker.epoch_update()
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    plot_training_history(loss_tracker)
    plt.show()


def evaluate_model(model_path: str, data_path: str):
    test_data = AirfoilDataset(os.path.join(data_path, 'test_airfoils.h5'))
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    model = model.to(device)
    model.eval()

    loss = MSELoss()
    losses = []

    for batch in test_loader:
        grid_x, grid_y, landmarks, u, v, rho, energy, mask, cd, cl, cm = batch
        landmarks = landmarks.flatten(start_dim=1).to(device)
        u = u.to(device)
        v = v.to(device)
        rho = rho.to(device)
        energy = energy.to(device)
        mask = mask.to(device)
        coefs = torch.cat((cl, cd, cm), 1).to(device)
        grid_x = grid_x.to(device)
        grid_y = grid_y.to(device)
        label = torch.stack((u, v, rho, energy), 1)

        pred_flow, _ = model.forward(grid_x, grid_y, landmarks, mask)

        losses.append(loss(y, label).item())

    print(f'Evaluation MSE: {np.mean(losses)}')


if __name__ == '__main__':
    model_dir = input('Model directory: ')
    data_dir = input('Data directory: ')
    train_model(model_dir, data_dir)
    evaluate_model(model_dir, data_dir)
