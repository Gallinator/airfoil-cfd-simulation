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
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, persistent_workers=True, num_workers=4)

    model = Model()
    model = model.to(device)
    model.train()

    flow_epochs = 120
    loss = MSELoss()

    flow_loss_tracker = LossTracker('Total')

    print('Training flow layers')
    flow_optimizer = AdamW(model.flow_parameters(), lr=0.0001)
    model.freeze_coefficients(True)
    for e in range(flow_epochs):
        prog = tqdm.tqdm(train_loader, f'Epoch {e}')
        for batch in prog:
            flow_optimizer.zero_grad()

            flow_data, _, flow_labels = batch
            flow_data = flow_data.to(device)
            flow_labels = flow_labels.to(device)

            pred_flow = model.flow_forward(flow_data)

            batch_loss = loss(flow_labels, pred_flow)
            batch_loss.backward()
            flow_optimizer.step()

            flow_loss_tracker.batch_update(Total=batch_loss.item())

        flow_loss_tracker.epoch_update()
        prog.write(f'Epoch {e} total loss: {flow_loss_tracker.loss_history['Total'][-1]}')
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    print('Training coefficients layers')
    coefs_loss_tracker = LossTracker('Total')
    coefs_epochs = 30
    coefs_optimizer = AdamW(model.coef_parameters(), lr=0.0001)
    model.freeze_airflow(True)
    model.freeze_coefficients(False)
    for e in range(coefs_epochs):
        prog = tqdm.tqdm(train_loader, f'Epoch {e}')
        for batch in prog:
            coefs_optimizer.zero_grad()

            flow_data, coef_labels, flow_labels = batch
            flow_data = flow_data.to(device)
            coef_labels = coef_labels.to(device)

            _, pred_coefs = model.forward(flow_data)

            batch_loss = loss(coef_labels, pred_coefs)
            batch_loss.backward()
            coefs_optimizer.step()

            coefs_loss_tracker.batch_update(Total=batch_loss.item())

        coefs_loss_tracker.epoch_update()
        prog.write(f'Epoch {e} total loss: {coefs_loss_tracker.loss_history['Total'][-1]}')
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    plot_training_history(flow_loss_tracker)
    plot_training_history(coefs_loss_tracker)
    plt.show()


def evaluate_model(model_path: str, data_path: str):
    eval_data = AirfoilDataset(os.path.join(data_path, 'test_airfoils.h5'))
    eval_loader = DataLoader(eval_data, batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    model = model.to(device)
    model.eval()

    loss = MSELoss()
    flow_losses = []
    coefs_losses = []

    for batch in tqdm.tqdm(eval_loader, 'Evaluating model'):
        flow_data, coef_labels, flow_labels = batch
        flow_data = flow_data.to(device)
        coef_labels = coef_labels.to(device)
        flow_labels = flow_labels.to(device)

        pred_flow, pred_coefs = model.forward(flow_data)

        flow_losses.append(loss(pred_flow, flow_labels).item())
        coefs_losses.append(loss(pred_coefs, coef_labels).item())

    flow_mean = np.mean(flow_losses)
    coefs_mean = np.mean(coefs_losses)
    print(f'Flow MSE: {flow_mean}')
    print(f'Coefficients MSE: {coefs_mean}')
    print(f'Total MSE: {flow_mean + coefs_mean}')


if __name__ == '__main__':
    model_dir = input('Model directory: ')
    data_dir = input('Data directory: ')
    train_model(model_dir, data_dir)
    evaluate_model(model_dir, data_dir)
