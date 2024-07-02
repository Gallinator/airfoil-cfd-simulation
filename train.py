import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from airfoil_dataset import AirfoilDataset
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


def get_diff_loss(out: torch.Tensor, loss):
    zeros = torch.zeros_like(out, requires_grad=True, dtype=torch.float32)
    return loss(out, zeros)


def train_model(save_path: str):
    train_data = AirfoilDataset('data/train_airfoils.h5')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8)

    in_size = train_data.landmarks[0].shape[0] * 2
    model = Model(len(train_data.grid_x), in_size)
    model = model.to(device)

    epochs = 40
    optimizer = AdamW(model.parameters(), lr=0.000001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=1e-4)
    loss = MSELoss()

    losses = []
    lr_history = []

    for e in range(epochs):
        epochs_losses = []
        prog = tqdm.tqdm(train_loader, desc=f'Epoch {e}')
        for batch in prog:
            optimizer.zero_grad()

            alpha, landmarks, u, v, p = batch
            landmarks = landmarks.flatten(start_dim=1).to(device)
            alpha = alpha.to(device)
            u = u.to(device)
            v = v.to(device)
            p = p.to(device)

            batch_size = landmarks.size()[0]
            grid_x = torch.tensor(train_data.grid_x, dtype=torch.float32).to(device).repeat(batch_size, 1)
            grid_y = torch.tensor(train_data.grid_y, dtype=torch.float32).to(device).repeat(batch_size, 1)

            pred_u, pred_v, pred_p, _, _, _ = model.forward(alpha, grid_x, grid_y, landmarks)

            batch_loss = loss(u, pred_u) + loss(v, pred_v) + loss(p, pred_p)
            batch_loss.backward()
            optimizer.step()

            epochs_losses.append(batch_loss.item())

        mean_epoch_loss = np.mean(epochs_losses)
        lr_history.append(optimizer.param_groups[0]['lr'])

        losses.append(mean_epoch_loss)
        scheduler.step(mean_epoch_loss)
        prog.write(f'Epoch {e} loss: {losses[-1]}')
        torch.save(model.state_dict(), save_path)

    plot_training_history(losses, lr_history)
    plt.show()


def evaluate_model(model_path: str):
    test_data = AirfoilDataset('data/test_airfoils.h5')
    test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=8)

    in_size = test_data.landmarks[0].shape[0] * 2
    model = Model(len(test_data.grid_x), in_size)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    loss = MSELoss()
    losses = []

    for batch in test_loader:
        alpha, landmarks, u, v, p = batch
        landmarks = landmarks.flatten(start_dim=1).to(device)
        alpha = alpha.to(device)
        u = u.to(device)
        v = v.to(device)
        p = p.to(device)

        batch_size = landmarks.size()[0]
        grid_x = torch.tensor(test_data.grid_x, dtype=torch.float32).to(device).repeat(batch_size, 1)
        grid_y = torch.tensor(test_data.grid_y, dtype=torch.float32).to(device).repeat(batch_size, 1)

        pred_u, pred_v, pred_p, _, _, _ = model.forward(alpha, grid_x, grid_y, landmarks)

        losses.append(loss(u, pred_u).item())
        losses.append(loss(v, pred_v).item())
        losses.append(loss(p, pred_p).item())

    print(f'Evaluation MSE: {np.mean(losses)}')


train_model('models/linear.pt')
evaluate_model('models/linear.pt')
