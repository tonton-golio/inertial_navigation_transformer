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
    
class GRUOrientation(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUOrientation, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        
        return out
    
class RNNOrientation(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNOrientation, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        
        return out
    
class CustomLoss(nn.Module):
    def __init__(self, dist_metric='phi2', agg_type='L2', normal=False):
        super().__init__()
        self.dist_metric = dist_metric
        self.agg_type = agg_type
        self.normal = normal

    def forward(self, q1, q2):
        if self.normal:
            q1 = self.normalize(q1)
            q2 = self.normalize(q2)
        if self.dist_metric == 'phi2':
            return self.phi2(q1, q2)
        elif self.dist_metric == 'phi4':
            return self.phi4(q1, q2)
        elif self.dist_metric == 'phi5':
            return self.phi5(q1, q2)
        else:
            raise ValueError('Invalid distance metric')

    def phi2(self, q1, q2):
        return self.aggregate(torch.min(torch.norm(q1 - q2), torch.norm(q1 + q2)))

    def phi4(self, q1, q2):
        return self.aggregate(1 - torch.abs(torch.einsum('ij,ij->i', q1, q2)))

    def phi5(self, q1, q2):
        R1 = self.quat_to_rot(q1)
        R2 = self.quat_to_rot(q2)
        return self.aggregate(torch.norm(torch.eye(3, device=q1.device)[None, :, :] - torch.bmm(R1, R2.transpose(-2, -1)), p='fro'))

    def quat_to_rot(self, q):
        q = self.normalize(q)

        q_r, q_i, q_j, q_k = q.split(1, dim=-1)
        q_r, q_i, q_j, q_k = q_r.squeeze(-1), q_i.squeeze(-1), q_j.squeeze(-1), q_k.squeeze(-1)

        R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device)
        R[..., 0, 0] = 1 - 2 * (q_j ** 2 + q_k ** 2)
        R[..., 0, 1] = 2 * (q_i * q_j - q_k * q_r)
        R[..., 0, 2] = 2 * (q_i * q_k + q_j * q_r)
        R[..., 1, 0] = 2 * (q_i * q_j + q_k * q_r)
        R[..., 1, 1] = 1 - 2 * (q_i ** 2 + q_k ** 2)
        R[..., 1, 2] = 2 * (q_j * q_k - q_i * q_r)
        R[..., 2, 0] = 2 * (q_i * q_k - q_j * q_r)
        R[..., 2, 1] = 2 * (q_j * q_k + q_i * q_r)
        R[..., 2, 2] = 1 - 2 * (q_i ** 2 + q_j ** 2)

        return R
    
    def normalize(self, q):
        return q / torch.norm(q, dim=-1, keepdim=True)

    def aggregate(self, x):
        if self.agg_type == 'L1':
            return torch.mean(torch.abs(x))
        elif self.agg_type == 'L2':
            return torch.mean(x ** 2)
        elif self.agg_type == 'log':
            return -torch.mean(torch.log(x))
        else:
            raise ValueError('Invalid aggregation type')


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

def objective_lstm(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 512, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    epochs = 200
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Adamax", "NAdam", "RAdam"])

    model = LSTMOrientation(input_size, hidden_size, num_layers, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = CustomLoss()
    
    if optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                               amsgrad=amsgrad, betas=(beta1, beta2))
    elif optimizer_name == "AdamW":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                amsgrad=amsgrad, betas=(beta1, beta2))
    elif optimizer_name == "Adamax":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == "NAdam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == "RAdam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    patience = 20
    min_valid_loss = np.inf
    patience_counter = 0

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
            print(f"Trial pruned with value: {val_loss.item()} at epoch {epoch} and parameters {trial.params} \n")
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1
        
        if patience_counter > patience:
            break
        
    return val_loss.item()

def objective_gru(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 512, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    epochs = 200
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Adamax", "NAdam", "RAdam"])

    model = GRUOrientation(input_size, hidden_size, num_layers, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = CustomLoss()
    
    if optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                               amsgrad=amsgrad, betas=(beta1, beta2))
    elif optimizer_name == "AdamW":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                amsgrad=amsgrad, betas=(beta1, beta2))
    elif optimizer_name == "Adamax":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == "NAdam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == "RAdam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    patience = 20
    min_valid_loss = np.inf
    patience_counter = 0

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
            print(f"Trial pruned with value: {val_loss.item()} at epoch {epoch} and parameters {trial.params} \n")
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1
        
        if patience_counter > patience:
            break
        
    return val_loss.item()

def objective_rnn(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 512, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    epochs = 200
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Adamax", "NAdam", "RAdam"])

    model = RNNOrientation(input_size, hidden_size, num_layers, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = CustomLoss()
    
    if optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                               amsgrad=amsgrad, betas=(beta1, beta2))
    elif optimizer_name == "AdamW":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                amsgrad=amsgrad, betas=(beta1, beta2))
    elif optimizer_name == "Adamax":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == "NAdam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == "RAdam":
        beta1 = trial.suggest_float("beta1", 0.0, 1.0)
        beta2 = trial.suggest_float("beta2", beta1, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    patience = 20
    min_valid_loss = np.inf
    patience_counter = 0

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
            print(f"Trial pruned with value: {val_loss.item()} at epoch {epoch} and parameters {trial.params} \n")
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1
        
        if patience_counter > patience:
            break
        
    return val_loss.item()


N_trials = 200

# to open the optuna dashboard run the following in a separate terminal
# $ optuna-dashboard sqlite:///optuna_optimizer.db
# then click on the http link to access the dashboard in your browser
study1 = optuna.create_study(direction="minimize", storage='sqlite:///optuna_optimizer.db', study_name='lstm_orientation_optimizer')
study1.optimize(objective_lstm, n_trials=N_trials, n_jobs=5)

print(study1.best_trial.params)

study2 = optuna.create_study(direction="minimize", storage='sqlite:///optuna_optimizer.db', study_name='gru_orientation_optimizer')
study2.optimize(objective_gru, n_trials=N_trials, n_jobs=5)

print(study2.best_trial.params)

study3 = optuna.create_study(direction="minimize", storage='sqlite:///optuna_optimizer.db', study_name='rnn_orientation_optimizer')
study3.optimize(objective_rnn, n_trials=N_trials, n_jobs=5)

print(study3.best_trial.params)
