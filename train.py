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

    in_size = train_data.landmarks[0].shape[0] * 2
    model = Model(len(train_data.grid_x), in_size)
    model = model.to(device)

    epochs = 40
    optimizer = AdamW(model.parameters(), lr=0.000001)
    loss = MSELoss()

    loss_tracker = LossTracker('total')

    for e in range(epochs):

        prog = tqdm.tqdm(train_loader, desc=f'Epoch {e}')
        for batch in prog:
            optimizer.zero_grad()

            alpha, landmarks, u, v, rho = batch
            landmarks = landmarks.flatten(start_dim=1).to(device)
            alpha = alpha.to(device)
            u = u.to(device)
            v = v.to(device)
            rho = rho.to(device)

            batch_size = landmarks.size()[0]
            grid_x = torch.tensor(train_data.grid_x, dtype=torch.float32).to(device).repeat(batch_size, 1)
            grid_y = torch.tensor(train_data.grid_y, dtype=torch.float32).to(device).repeat(batch_size, 1)

            pred_u, pred_v, pred_rho = model.forward(alpha, grid_x, grid_y, landmarks)

            batch_loss = loss(u, pred_u) + loss(v, pred_v) + loss(rho, pred_rho)
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

    in_size = test_data.landmarks[0].shape[0] * 2
    model = Model(len(test_data.grid_x), in_size)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    loss = MSELoss()
    losses = []

    for batch in test_loader:
        alpha, landmarks, u, v, rho = batch
        landmarks = landmarks.flatten(start_dim=1).to(device)
        alpha = alpha.to(device)
        u = u.to(device)
        v = v.to(device)
        rho = rho.to(device)

        batch_size = landmarks.size()[0]
        grid_x = torch.tensor(test_data.grid_x, dtype=torch.float32).to(device).repeat(batch_size, 1)
        grid_y = torch.tensor(test_data.grid_y, dtype=torch.float32).to(device).repeat(batch_size, 1)

        pred_u, pred_v, pred_rho = model.forward(alpha, grid_x, grid_y, landmarks)

        losses.append(loss(u, pred_u).item())
        losses.append(loss(v, pred_v).item())
        losses.append(loss(rho, pred_rho).item())

    print(f'Evaluation MSE: {np.mean(losses)}')


train_model('models/linear.pt')
evaluate_model('models/linear.pt')
