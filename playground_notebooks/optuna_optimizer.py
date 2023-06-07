import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import itertools
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split

from ahrs.filters.madgwick import Madgwick
from ahrs.filters.mahony import Mahony
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q

file_path = '/home/asp/Downloads/AML/final-project/a000_1/data.hdf5'

if '..' not in sys.path:
    sys.path.append('../')
sys.path.append('inertial_navigation_transformer')
from utils import load_data
data_dict = load_data(file_path, verbose=False)

N = 10000

if N == 0:
    ekf_ori = data_dict['pose/ekf_ori']
    tango_ori = data_dict['pose/tango_ori']
    gyro = data_dict['synced/gyro']
    gyro_uncalib = data_dict['synced/gyro_uncalib']
    acc = data_dict['synced/acce']
    linacc = data_dict['synced/linacce']
    mag = data_dict['synced/magnet']
    num_samples = ekf_ori.shape[0]
    time = data_dict['synced/time']
    rv = data_dict['synced/rv']
    game_rv = data_dict['synced/game_rv']
    diffs = np.diff(time)
    dt = np.mean(diffs)
    q_0 = ekf_ori[0,:]

else:
    ekf_ori = data_dict['pose/ekf_ori'][:N]
    tango_ori = data_dict['pose/tango_ori'][:N]
    gyro = data_dict['synced/gyro'][:N]
    gyro_uncalib = data_dict['synced/gyro_uncalib'][:N]
    acc = data_dict['synced/acce'][:N]
    linacc = data_dict['synced/linacce'][:N]
    mag = data_dict['synced/magnet'][:N]
    num_samples = ekf_ori.shape[0]
    time = data_dict['synced/time'][:N]
    rv = data_dict['synced/rv'][:N]
    game_rv = data_dict['synced/game_rv'][:N]
    diffs = np.diff(time)
    dt = np.mean(diffs)
    q_0 = ekf_ori[0,:]


gain = 1.12666016
madgwick_filter = Madgwick(gyr = gyro, acc = acc, mag=mag, dt = dt, q0=q_0, gain=gain)
Q_1 = madgwick_filter.Q

k_P = 0.5366640716587453
k_I = 0.4086550384556218
mahony_filter = Mahony(gyr = gyro, acc = acc, mag=mag, dt = dt, q0=q_0, k_P=k_P, k_I=k_I)
Q_2 = mahony_filter.Q

acc_var = 0.3**2
gyro_var = 0.5**2
mag_var = 0.8**2
ekf = EKF(gyr=gyro, acc=acc, mag=mag, dt = dt, q0=q_0, noises= [gyro_var, acc_var, mag_var])
Q_ekf = ekf.Q


if not torch.is_tensor(gyro):
    gyro = torch.from_numpy(gyro).float()
if not torch.is_tensor(acc):
    acc = torch.from_numpy(acc).float()
if not torch.is_tensor(mag):
    mag = torch.from_numpy(mag).float()
if not torch.is_tensor(rv):
    rv = torch.from_numpy(rv).float()
if not torch.is_tensor(game_rv):
    game_rv = torch.from_numpy(game_rv).float()
if not torch.is_tensor(Q_1):
    Q_1 = torch.from_numpy(Q_1).float()
if not torch.is_tensor(Q_2):
    Q_2 = torch.from_numpy(Q_2).float()
if not torch.is_tensor(Q_ekf):
    Q_ekf = torch.from_numpy(Q_ekf).float()
if not torch.is_tensor(tango_ori):
    tango_ori = torch.from_numpy(tango_ori).float()


class LSTMOrientation(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMOrientation, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :])
        
        return out
    
class CustomLoss(nn.Module):
    def __init__(self, dist_metric='norm', agg_type='L1'):
        super().__init__()
        self.dist_metric = dist_metric
        self.agg_type = agg_type

    def forward(self, q1, q2):
        if self.dist_metric == 'qdist1':
            loss = torch.min(torch.abs(torch.sum(q1 - q2, dim=-1)), torch.abs(torch.sum(q1 + q2, dim=-1)))
        elif self.dist_metric == 'qdist2':
            cos_half_angle = torch.abs(torch.sum(q1 * q2, dim=-1))
            loss = 2 * torch.acos(torch.clamp(cos_half_angle, -1.0, 1.0))
        elif self.dist_metric == 'qdist3':
            loss = 1 - torch.abs(torch.sum(q1 * q2, dim=-1))
        elif self.dist_metric == 'norm':
            loss = torch.norm(q1 - q2, dim=-1)
        else:
            raise ValueError(f'Invalid dist_metric: {self.dist_metric}')

        if self.agg_type == 'L1':
            return torch.mean(torch.abs(loss))  # L1 loss (MAE)
        elif self.agg_type == 'L2':
            return torch.mean(torch.pow(loss, 2))  # L2 loss (MSE)
        else:
            raise ValueError(f'Invalid agg_type: {self.agg_type}')


input_size = 21
hidden_size = 128
num_layers = 3
output_size = 4
seq_len = 5

train_data = torch.cat((gyro, acc, mag, Q_1, Q_2, Q_ekf), dim=1)

input_size = train_data.shape[1]

train_data = train_data.view(num_samples, -1, input_size)
train_labels = tango_ori

val_size = int(0.2 * len(train_data))
train_size = num_samples - val_size

val_data, train_data = train_data[-val_size:], train_data[:-val_size]
val_labels, train_labels = train_labels[-val_size:], train_labels[:-val_size]

train_data = train_data.view(train_size, -1, input_size)

def objective(trial):
    epochs = trial.suggest_int("epochs", 10, 200)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", 
                                                             "Adagrad", "AdamW", 
                                                             "Adamax", "ASGD", "NAdam", "RAdam"])

    model = LSTMOrientation(input_size, hidden_size, num_layers, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = CustomLoss(dist_metric='norm', agg_type='L1')
    
    if optimizer_name == "Adam":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                               amsgrad=False)
    elif optimizer_name == "SGD":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 1e-6, 1e-1, log=True)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                              momentum=momentum, nesterov=True)
    elif optimizer_name == "Adagrad":
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        lr_decay = trial.suggest_float("lr_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                                  lr_decay=lr_decay)
    elif optimizer_name == "AdamW":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                amsgrad=amsgrad)
    elif optimizer_name == "Adamax":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "ASGD":
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        lambd = trial.suggest_float("lambd", 1e-6, 1e-3, log=True)
        optimizer = optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                               lambd=lambd)
    elif optimizer_name == "NAdam":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        momentum_decay = trial.suggest_float("momentum_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                                momentum_decay=momentum_decay)
    elif optimizer_name == "RAdam":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        outputs = model(train_data.to(device))
        loss = criterion(outputs, train_labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data.to(device))
            val_loss = criterion(val_outputs, val_labels.to(device))
            
        trial.report(val_loss.item(), epoch)

        if trial.should_prune() and epoch > 5:
            print(f"Trial pruned with value: {val_loss.item()} at epoch {epoch} and parameters {trial.params}")
            raise optuna.exceptions.TrialPruned()
        
    return val_loss.item()

# to open the optuna dashboard run the following in a separate terminal
# $ optuna-dashboard sqlite:///optuna_optimizer.db
# then click on the http link to access the dashboard in your browser
study = optuna.create_study(direction="minimize", storage='sqlite:///optuna_optimizer.db')
study.optimize(objective, n_trials=100, n_jobs=5)

print(study.best_trial.params)
