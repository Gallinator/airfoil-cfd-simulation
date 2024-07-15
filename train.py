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

    loss_tracker = LossTracker('Total', 'Coefficients', 'Flow')

    for e in range(epochs):

        prog = tqdm.tqdm(train_loader, f'Epoch {e}')
        for batch in prog:
            optimizer.zero_grad()

            flow_data, coef_labels, flow_labels = batch
            flow_data = flow_data.to(device)
            coef_labels = coef_labels.to(device)
            flow_labels = flow_labels.to(device)

            pred_flow, pred_coefs = model.forward(flow_data)

            coefs_loss = loss(coef_labels, pred_coefs)
            air_loss = loss(flow_labels, pred_flow)

            batch_loss = air_loss + coefs_loss * 0.75
            batch_loss.backward()
            optimizer.step()

            loss_tracker.batch_update(Total=batch_loss.item(), Coefficients=coefs_loss.item(), Flow=air_loss.item())

        loss_tracker.epoch_update()
        prog.write(f'Epoch {e} flow loss: {loss_tracker.loss_history['Flow'][-1]}')
        prog.write(f'Epoch {e} total loss: {loss_tracker.loss_history['Total'][-1]}')
        prog.write(f'Epoch {e} coefs loss: {loss_tracker.loss_history['Coefficients'][-1]}')
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    plot_training_history(loss_tracker)
    plt.show()


def evaluate_model(model_path: str, data_path: str):
    eval_data = AirfoilDataset(os.path.join(data_path, 'test_airfoils.h5'))
    eval_loader = DataLoader(eval_data, batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    model = model.to(device)
    model.eval()

    loss = MSELoss()
    losses = []

    for batch in tqdm.tqdm(eval_loader, 'Evaluating model'):
        flow_data, coef_labels, flow_labels = batch
        flow_data = flow_data.to(device)
        coef_labels = coef_labels.to(device)
        flow_labels = flow_labels.to(device)

        pred_flow, pred_coefs = model.forward(flow_data)

        losses.append(loss(pred_flow, flow_labels).item())
        losses.append(loss(pred_coefs, coef_labels).item())

    print(f'Evaluation MSE: {np.mean(losses)}')


if __name__ == '__main__':
    model_dir = input('Model directory: ')
    data_dir = input('Data directory: ')
    train_model(model_dir, data_dir)
    evaluate_model(model_dir, data_dir)
