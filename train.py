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
from visualization import plot_training_history


def get_device():
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Using CPU')
    else:
        print('CUDA is available!  Using GPU')

    return torch.device("cuda:0" if train_on_gpu else "cpu")


device = get_device()


def train_model(save_path: str):
    train_data = AirfoilDataset('data/train_airfoils.h5')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    model = model.to(device)
    model.train()

    epochs = 40
    optimizer = AdamW(model.parameters(), lr=0.000001)
    loss = MSELoss()

    loss_tracker = LossTracker('total')

    for e in range(epochs):

        prog = tqdm.tqdm(train_loader, desc=f'Epoch {e}')
        for batch in prog:
            optimizer.zero_grad()

            grid_x, grid_y, landmarks, u, v, rho, energy, mask = batch
            landmarks = landmarks.flatten(start_dim=1).to(device)
            grid_x = grid_x.to(device)
            grid_y = grid_y.to(device)
            u = u.to(device)
            v = v.to(device)
            rho = rho.to(device)
            energy = energy.to(device)
            mask = mask.to(device)
            label = torch.stack((u, v, rho, energy), 1)

            y = model.forward(grid_x, grid_y, landmarks, mask)

            batch_loss = loss(label, y)
            batch_loss.backward()
            optimizer.step()

            loss_tracker.batch_update(total=batch_loss.item())

        loss_tracker.epoch_update()
        prog.write(f'Epoch {e} loss: {loss_tracker.loss_history['total'][-1]}')
        torch.save(model.state_dict(), save_path)

    plot_training_history(loss_tracker)
    plt.show()


def evaluate_model(model_path: str):
    test_data = AirfoilDataset('data/test_airfoils.h5')
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    loss = MSELoss()
    losses = []

    for batch in test_loader:
        grid_x, grid_y, landmarks, u, v, rho, energy, mask = batch
        landmarks = landmarks.flatten(start_dim=1).to(device)
        u = u.to(device)
        v = v.to(device)
        rho = rho.to(device)
        energy = energy.to(device)
        mask = mask.to(device)
        grid_x = grid_x.to(device)
        grid_y = grid_y.to(device)
        label = torch.stack((u, v, rho, energy), 1)

        y = model.forward(grid_x, grid_y, landmarks, mask)

        losses.append(loss(y, label).item())

    print(f'Evaluation MSE: {np.mean(losses)}')


train_model('models/linear.pt')
evaluate_model('models/linear.pt')
