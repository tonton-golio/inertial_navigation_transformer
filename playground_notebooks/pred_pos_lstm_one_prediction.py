# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import pickle
import time

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler

import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ahrs.filters.madgwick import Madgwick
from ahrs.filters.mahony import Mahony
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure
sys.path.append('inertial_navigation_transformer')

from utils import load_much_data, load_split_data, create_sequences, load_data, normalize_features

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)

# set data folder path
folder_path1 = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1'
folder_path_full = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_full'
cluster_labels_path = 'C:\\Users\\Simon Andersen\\Projects\\Projects\\inertial_navigation_transformer\\Clustering_labels'
test_set_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\unseen_subjects_test_set'

### FUNCTIONS ----------------------------------------------------------------------------------

class Net(nn.Module):
          def __init__(self, model, device, input_size, batch_size=64, hidden_size=7, num_layers = 1,\
                        output_size = 4, seq_len = 1, dropout = 0):
               super().__init__()

               self.hidden_size = hidden_size
               self.num_layers = num_layers
               self.batch_size = batch_size
               self.lstm_size = hidden_size
               self.device = device

               self.lstm = model(input_size=input_size, hidden_size=hidden_size, \
                                   num_layers = num_layers, batch_first=True, dropout = dropout)
               #self.relu = nn.ReLU()
               self.linear1 = nn.Linear(hidden_size, 5 * hidden_size)
               self.linear2 = nn.Linear(5 * hidden_size, output_size)
                    
          def forward(self, x):
               #h0 = torch.zeros(self.num_layers *2, x.size(0), self.hidden_size).to(x.device)
               #c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

               x, states = self.lstm(x)  # Run LSTM and store the hidden layer outputs
               x = x[:, -1, :]  # take the last hidden layer
               #x = self.relu(x) # a normal dense layer
               x = self.linear1(x)
               x = self.linear2(x)
               return x
          
          def init_weights(self):
               h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
               c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
               h0 = h0.to(self.device)
               c0 = c0.to(self.device)
               return torch.autograd.Variable(h0), torch.autograd.Variable(c0)



### MAIN ---------------------------------------------------------------------------------------

## TODO
# Make plotting pretty
# set up proper testing on some unseen data
# Fig. out why it worsens with more epochs. Play around with architechture 
# Take steps to improve performance. Initializatoin, regularziation etc. Look at guide
# Implement a hyper parameter optimization scheme
# Do large scale hyper parameter optimization

def main():
     ### Implement LSTM
     model_name = 'rnn'
     architecture = nn.RNN
     n_trial = 0
     ## Set central parameters
    # set whether to include clustered labels
     load_model, do_training, do_simple_testing, do_many_tests, do_testing, plot_testing, do_hyperparameter_search \
          = False, False, False, False, False, False, True
     include_clusters = False

     # set whether to predict only the last value per sequence
     pred_only_last = True
     # set whether to test on same dataset (but unseen points)
     test_on_same_dataset = False
     plot_train_val_predictions = False
     shuffle = True
     
     # ... and to include the previous test prediction as the input when testing net batch
     use_truth_input = False
     # set whether to predict 2 or 3D positions
     pred_position_in_2D = True

     # Set how many observations to combine into one datapoint
     Ntrain = 1_200_000 # 2_400_000 # 1_200_000 #2_800_000  #3_000_000 + 1_500_000
     Ntest = 67_000
     num_datasets = max(1, int(Ntrain / 60_000)) # int(Ntrain / 60_000)
     folder_path = folder_path1 if num_datasets < 50 else folder_path_full
     random_start = False
     
     # Set whether to include acceleration and magnetic data when training 
     include_acc = False
     include_lin_acc = True
     include_mag = True
     # include truth ori in preicting positions
     include_tango_ori  = True
     include_ekf_ori = False
     normalize = True
     # set whether to include truth data when training. 
     include_filtered_features = False
     include_madgwick = True
     include_mahony = True
     include_EKF = True
     # set whether to include rv and game rv 
     include_rv = False
     include_game_rv = False
     include_gps = False
     
     {'epochs': 117, 'hidden_size': 74, 'learning_rate': 0.00017326141536061154, 'num_layers': 3, 'seq_len': 35}

     learning_rate = 8e-5
     # Set loss function
     criterion = nn.MSELoss()
     # set n_epochs

     overlap = 1
     # set sequence length
     seq_len = 30
     dt = 1 / 200

     num_epochs = 35  
      # set size of hidden layers
     hidden_size = 158
     # set batch size
     batch_size = 64
     # set no. of layers in LSTM
     num_layers = 3
     # set the dropout layer strength
     dropout_layer = 0.2

     Ntest_fraction = Ntest / Ntrain
     # Decide which fraction of the training data should be used for validation
     Nval_fraction = .2

     output_features = ['pose/tango_pos']
     input_features = ['synced/gyro']
     if include_acc:
          input_features.append('synced/acce')
     if include_lin_acc:
          input_features.append('synced/linacce')
     if include_mag:
          input_features.append('synced/magnet')
     if include_tango_ori:
          input_features.append('pose/tango_ori')
     if include_ekf_ori:
          input_features.append('pose/ekf_ori')
     if include_rv: 
          input_features.append('synced/rv')
     if include_game_rv:
          input_features.append('synced/game_rv')
     if include_gps:
          input_features.append('raw/tango/gps')
     if use_truth_input:
          input_features.append(output_features[0])
     
     params = {'N_train': Ntrain, 
               'N_test': Ntest,
               'seq_len': seq_len, 
               'input': input_features,
               'output': output_features, 
               'normalize': normalize,
               'verbose': False, 'num_datasets':num_datasets,
               'random_start':random_start,
               'overlap': overlap,
               'include_clusters': include_clusters,
               'cluster_labels_path': cluster_labels_path}
     

     X_train, y_train, X_test, y_test, col_pos, scaler = load_split_data(folder_path=folder_path,\
                                                                   return_scaler=True, **params)

     X_test_copy = X_test.astype('float')
     print(col_pos)
     
     print('Xtrain', X_train.shape, 'ytrain', y_train.shape)
     print('Xtest', X_test.shape, 'ytest', y_test.shape)

     if test_on_same_dataset:

          X_test = X_train[-int(Ntest/seq_len):]
          y_test = y_train[-int(Ntest/seq_len):]
          X_train = X_train[:-int(Ntest/seq_len)]
          y_train = y_train[:-int(Ntest/seq_len)]
     if include_filtered_features:
          Ntrain_sequences = X_train.shape[0]
        
          assert(include_mag is True & (include_acc is True if include_lin_acc is False else False))
          assert(input_features[0] == 'synced/gyro')
          assert(input_features[1] == 'synced/acce' if include_acc else 'synced/linacce')
          assert(input_features[2] == 'synced/magnet')
     
          gyro_idx = np.arange(3)
          acc_idx = np.arange(3,6)
          mag_idx = np.arange(6,9)
      

          X = np.vstack((X_train, X_test))
          q_0 = y_train[0, 0, :]

          # Extract rows, reshape to (Npoints, Nfeatures) 
          X_gyro = X[:, :, gyro_idx].reshape(-1, 3)
          X_acc = X[:, :, acc_idx].reshape(-1, 3)
          X_mag = X[:, :, mag_idx].reshape(-1, 3)
          dt = 1/200

          print(X_gyro.shape, X_acc.shape, X_mag.shape)

          if include_madgwick:
               gain = 1.04299724
               madgwick_filter = Madgwick(gyr = X_gyro, acc = X_acc, mag=X_mag, dt = dt, q0=q_0, gain=gain)
               Q_1 = madgwick_filter.Q
               # Add now features to X
               print(Q_1.shape)
               print(Q_1.reshape(-1, seq_len, 4).shape)
               X = np.concatenate((Q_1.reshape(-1, seq_len, 4), X), axis=-1)

          if include_mahony:
               k_P = 0.8311948880973625
               k_I = 0.13119070597985183
               mahony_filter = Mahony(gyr = X_gyro, acc = X_acc, mag=X_mag, dt = dt, q0=q_0, k_P=k_P, k_I=k_I)
               Q_2 = mahony_filter.Q
               X = np.concatenate((Q_2.reshape(-1, seq_len, 4), X), axis=-1)

          
          if include_EKF:
               acc_var = 0.3**2
               gyro_var = 0.5**2
               mag_var = 0.8**2
               ekf = EKF(gyr=X_gyro, acc = X_acc, mag=X_mag, dt = dt, q0=q_0, noises= [gyro_var, acc_var, mag_var])
               Q_ekf = ekf.Q
               X = np.concatenate((Q_ekf.reshape(-1, seq_len, 4), X), axis=-1)

          X_train = X[:Ntrain_sequences, :, :]    
          X_test = X[Ntrain_sequences:, :, :]

          if 0:
               for q_0, X in zip([y_train[0, 0, :], y_train[0, 0, :]], [X_train, X_test]):
                    # Extract rows, reshape to (Npoints, Nfeatures) 
                    X_gyro = X[:, :, gyro_idx].reshape(-1, 3)
                    X_acc = X[:, :, acc_idx].reshape(-1, 3)
                    X_mag = X[:, :, mag_idx].reshape(-1, 3)
                    dt = 1/200
     
                    print(X_gyro.shape, X_acc.shape, X_mag.shape)

                    if include_madgwick:
                         gain = 1.04299724
                         madgwick_filter = Madgwick(gyr = X_gyro, acc = X_acc, mag=X_mag, dt = dt, q0=q_0, gain=gain)
                         Q_1 = madgwick_filter.Q
                         # Add now features to X
                         print(Q_1.shape)
                         print(Q_1.reshape(-1, seq_len, 4).shape)
                         X = np.concatenate((Q_1.reshape(-1, seq_len, 4), X), axis=-1)

                    if include_mahony:
                         k_P = 0.8311948880973625
                         k_I = 0.13119070597985183
                         mahony_filter = Mahony(gyr = X_gyro, acc = X_acc, mag=X_mag, dt = dt, q0=q_0, k_P=k_P, k_I=k_I)
                         Q_2 = mahony_filter.Q
                         X = np.concatenate((Q_2.reshape(-1, seq_len, 4), X), axis=-1)

                    
                    if include_EKF:
                         acc_var = 0.3**2
                         gyro_var = 0.5**2
                         mag_var = 0.8**2
                         ekf = EKF(gyr=X_gyro, acc = X_acc, mag=X_mag, dt = dt, q0=q_0, noises= [gyro_var, acc_var, mag_var])
                         Q_ekf = ekf.Q
                         X = np.concatenate((Q_ekf.reshape(-1, seq_len, 4), X), axis=-1)

          print("Shape of Xtrain, Xtest after including filtered features", X_train.shape, X_test.shape)

     print('Xtrain', X_train.shape, 'ytrain', y_train.shape)
     print('Xtest', X_test.shape, 'ytest', y_test.shape)
     
     Noutput_features = y_train.shape[-1]
     # Set all but the first true value to 0 in each sequence
     if use_truth_input:
          X_train[:, 1:, -Noutput_features:] = 0
          X_test[:, 1:, -Noutput_features:] = 0
     
     # predict only in 2D
     y_train = y_train[:, :, :2]
     y_test = y_test[:, :, :2]

     ## Prepare and split train data into train and val. data
     X_train_arr, X_val_arr, y_train_arr, y_val_arr = \
          train_test_split(X_train, y_train, \
                                                       test_size=Nval_fraction, shuffle=False)
    
     # shift all predictions for each each sequence to start at 0
     y_train_offset = y_train_arr[:, 0, :]
     y_val_offset = y_val_arr[:,0,:]
     y_test_offset = y_test[:, 0, :]
     y_train_arr = y_train_arr - y_train_arr[:,0,:][:, np.newaxis, :]
     y_val_arr = y_val_arr - y_val_arr[:,0,:][:, np.newaxis, :]
     y_test_copy = y_test.astype('float')
     y_test = y_test - y_test[:,0,:][:, np.newaxis, :]

     print('Xtrain', X_train_arr.shape, 'ytrain', y_train_arr.shape)
     print('Xval', X_val_arr.shape, 'yval', y_val_arr.shape)
     print('Xtest', X_test.shape, 'ytest', y_test.shape)
     
     if pred_only_last:
          # Include only the last true value in each sequence
          y_train_arr = y_train_arr[:,-1,:]#.reshape(y_train.shape[0], - 1)
          y_val_arr = y_val_arr[:,-1,:]#.reshape(y_val.shape[0], - 1)
          y_test = y_test[:,-1,:]#.reshape(y_test.shape[0], - 1)

     print('Xtrain', X_train.shape, 'ytrain', y_train_arr.shape)
     print('Xtest', X_test.shape, 'ytest', y_test.shape)

     output_size = y_train_arr.shape[-1]
     Noutput_features = output_size
     Nfeatures = X_train.shape[-1]


     print(X_train_arr.shape, X_val_arr.shape, y_train_arr.shape, y_val_arr.shape)

     # convert to torch tensors
     X_train = torch.from_numpy(X_train_arr).float()
     y_train = torch.from_numpy(y_train_arr).float()
     X_val = torch.from_numpy(X_val_arr).float()
     y_val = torch.from_numpy(y_val_arr).float()
     X_test = torch.from_numpy(X_test).float()
     y_test = torch.from_numpy(y_test).float()

     print("tensor shape: ", y_val.shape, y_train.shape, y_test.shape)

     # dataset
     train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
     val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
     test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

     # dataloader
     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


     # setting device on GPU if available, else CPU
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print('Using device:', device)

     #Additional Info when using cuda
     if device.type == 'cuda':
          print(torch.cuda.get_device_name(0))
          print('Memory Usage:')
          print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
          print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



     net = Net(architecture, device, input_size=Nfeatures, batch_size=batch_size, output_size = output_size,hidden_size=hidden_size,\
                num_layers=num_layers,  seq_len=seq_len, dropout=dropout_layer)
     # mount model to device
     net.to(device)
     # number of params
     num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
     print('Number of parameters: %d' % num_params)
       
     # optimizer
     optimizer = optim.Adam(net.parameters(), lr=learning_rate)
     t_start = time.time()

     if do_training:
          # TRAIN -----------------------------------------------------------
          for epoch in range(num_epochs):
               running_loss = 0.0
               n_minibatches = 0
               try:
                    for inputs, labels in train_loader:
                         inputs, labels = inputs.to(device), labels.to(device)
                         optimizer.zero_grad() 
                         outputs = net(inputs)
                         
                         loss = criterion(outputs, labels)
                         loss.backward()
                         optimizer.step()
                         running_loss += loss.item()
                         n_minibatches += 1
                    print('Epoch %d, loss: %.5f' % (epoch+1, running_loss/n_minibatches))

                    # Validate
                    with torch.no_grad():
                         running_loss = 0.0
                         n_minibatches = 0
                         for inputs, labels in val_loader:
                              inputs, labels = inputs.to(device), labels.to(device)
                              outputs = net(inputs)
                         
                              loss = criterion(outputs, labels)
                              running_loss += loss.item()
                              n_minibatches += 1
                         print('Test loss: %.5f' % (running_loss/n_minibatches))
               except KeyboardInterrupt:
                    print('Interrupted by user at epoch %d' % (epoch+1))
                    break

          # save model
          torch.save(net.state_dict(), f'best{model_name}{n_trial}.pt')
          t_end = time.time()
          print("Training time: ", t_end - t_start)

     if do_hyperparameter_search:
          # Hyperparameter search -----------------------------------------------------------
          model_name_list = ['gru', 'rnn']
          architecture_list = [nn.GRU, nn.RNN]

          d_gru = {'epochs': (30,150), 'learning_rate': (5e-6, 5e-4), 'num_layers': (3,3), 'hidden_size': (40,100), 'seq_len': (5,50)}
          d_rnn = {'epochs': (30,150), 'learning_rate': (5e-6, 10e-4), 'num_layers': (3,5), 'hidden_size': (40,160), 'seq_len': (5,50)}
          
          d_list = [d_gru, d_rnn]  

          for i,architecture in enumerate([architecture_list[1]]):

   
               def objective(trial, model):
                    num_epochs = trial.suggest_int("epochs", d_list[i]['epochs'][0], d_list[i]['epochs'][1])
                    learning_rate = trial.suggest_float("learning_rate", d_list[i]['learning_rate'][0], d_list[i]['learning_rate'][1], log=False)
                    hidden_size = trial.suggest_int("hidden_size", d_list[i]['hidden_size'][0], d_list[i]['hidden_size'][1]) 
                    num_layers = trial.suggest_int("num_layers", d_list[i]['num_layers'][0], d_list[i]['num_layers'][1])     
                    seq_len = trial.suggest_int("seq_len", d_list[i]['seq_len'][0], d_list[i]['seq_len'][1])  

                    if 0:
                         learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=False)
                         hidden_size = trial.suggest_int("hidden_size", 40, 100)
                         num_layers = trial.suggest_int("num_layers", 1, 3)
                         seq_len = trial.suggest_int("seq_len", 5, 50)


                    model = Net(model, device, input_size=Nfeatures, output_size = output_size,hidden_size=hidden_size,\
                    num_layers=num_layers,  seq_len=seq_len, dropout=dropout_layer)
                    model.to(device)
                    optimizer = optim.Adam(net.parameters(), lr=learning_rate,)
                    criterion = nn.MSELoss()
                    # TRAIN -----------------------------------------------------------
                    val_loss = np.empty(num_epochs)
                    
                    for epoch in range(num_epochs):
                         model.train()
                         for inputs, labels in train_loader:
                              inputs, labels = inputs.to(device), labels.to(device)
                              optimizer.zero_grad() 
                              outputs = model(inputs)
                              
                              loss = criterion(outputs, labels)
                              loss.backward()
                              optimizer.step()

                         # Validate
                         model.eval()
                         with torch.no_grad():
                              running_loss = 0.0
                              n_minibatches = 0
                              for inputs, labels in val_loader:
                                   inputs, labels = inputs.to(device), labels.to(device)
                                   outputs = model(inputs)
                              
                                   loss = criterion(outputs, labels)
                                   running_loss += loss.item()
                                   n_minibatches += 1
                         trial.report(running_loss/n_minibatches, epoch)
                         if trial.should_prune() and epoch > 25:
                              print(f"Trial pruned with value: {running_loss/n_minibatches} at epoch {epoch} and parameters {trial.params}")
                              raise optuna.exceptions.TrialPruned()
                         val_loss[epoch] = running_loss/n_minibatches

                    # store model
                    #trial.set_user_attr(key="model", value=model)
                    return val_loss[-5:].mean()

               def callback(study, trial):
                    if study.best_trial.number == trial.number:
                         study.set_user_attr(key="best_model", value=trial.user_attrs["model"])

               # to open the optuna dashboard run the following in a separate terminal
               # $ optuna-dashboard sqlite:///optuna_optimizer.db
               # then click on the http link to access the dashboard in your browser
     
               # create optimization object
               study = optuna.create_study(direction="minimize",sampler=TPESampler(),\
                                        pruner=MedianPruner(n_warmup_steps=50), storage='sqlite:///optuna_optimizer.db')

               # create objective function to pass and optimize
               restricted_objective = lambda trial: objective(trial, architecture)
               study.optimize(restricted_objective, n_trials=40, show_progress_bar=False)  #callbacks=[callback]
          
               print("Best params: ", study.best_trial.params)
               print("Best loss: ", study.best_trial.values)

               # Save best parameters
               with open(f'best_params_{model_name_list[i]}.pkl', 'wb') as fp:
                    pickle.dump(study.best_trial.params, fp)


     if load_model:
          # Load params
          # ....
          # load model
          net = Net(architecture, device, input_size=Nfeatures, output_size = output_size,hidden_size=hidden_size,\
               num_layers=num_layers,  seq_len=seq_len, dropout=dropout_layer)
          if n_trial is not None:
               net.load_state_dict(torch.load(f'best{model_name}{n_trial}.pt'))
          else:
               net.load_state_dict(torch.load(f'best{model_name}.pt'))
          net.to(device)
          net.eval()


     if do_simple_testing:

          ## TEST ------------------------------------------------------
          y_pred = []
          pred_list = [y_pred]
          loaders = [test_loader]
          if plot_train_val_predictions:
               train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
               val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
               y_pred_train = []
               y_pred_val = []  
               pred_list = [y_pred, y_pred_train, y_pred_val]
               loaders = [test_loader, train_loader, val_loader]
          for j, pred in enumerate(pred_list):
               if use_truth_input:
                    with torch.no_grad():
                         for i, (inputs, labels) in enumerate(loaders[j]):
                              inputs, labels = inputs.to(device), labels.to(device)
                              if i > 10:
                                        # copy last prediction to input
                                        inputs[:,0,-Noutput_features:] = y_pred_tensor[0]

                              outputs = net(inputs)
                              # copy to cpu and store prediction
                              outputs_cpu = torch.Tensor.cpu(outputs)
                              pred.append(outputs_cpu.numpy())
                              y_pred_tensor = outputs
               else:
                    with torch.no_grad():
                         for i, (inputs, labels) in enumerate(loaders[j]):
                              inputs, labels = inputs.to(device), labels.to(device)
                    
                              outputs = net(inputs)    
                              # copy to cpu and store prediction
                              outputs_cpu = torch.Tensor.cpu(outputs)
                              pred.append(outputs_cpu.numpy())
                              y_pred_tensor = outputs
          

          nrows = 1 if Noutput_features == 3 else 2
          ncols = 3 if Noutput_features == 3 else 2

          fig2, ax2 = plt.subplots()
          pred_true = [y_test, y_train_arr, y_val_arr]
          pred_offset = [y_test_offset, y_train_offset, y_val_offset]
          name_list = ['Test', 'Train', 'Validation']
          marker_list = ['-','--','--']
          alpha_list = [0.8,0.4,0.4]
          color_list_true = ['green', 'olive', 'green']
          color_list_pred = ['red', 'plum', 'red']

          for i, pred in enumerate(pred_list):

     
               y_pred = np.array(pred)
               y_pred = y_pred.reshape(-1, Noutput_features)
               y_pred = y_pred + np.vstack([pred_offset[i][0,:], pred_offset[i][0,:] + np.cumsum(y_pred[:-1], axis=0)])
               # start predictions at the initial ground truth position
               y_pred = np.vstack([pred_offset[i][0,:], y_pred])

               # save predictions
               np.savetxt(f'predictions_arr{trial_no}.txt', y_pred, delimiter=',')

               # Include only as many test points as is divisible by batch size
               q_gts = pred_true[i].reshape(-1, Noutput_features)
               q_gts = q_gts + pred_offset[i]
               q_gts = np.vstack([pred_offset[i][0,:], q_gts])
               q_preds = y_pred
               xs = np.arange(len(q_gts))

               dt = 1 / 200
               ATE_vec = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1))
               ATE = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1)).mean()
               Nseconds = Ntest * dt
               Nmins = int(np.floor(Nseconds / 60))
               minute_idx = int(Ntest * 60 / Nseconds / seq_len)
               RTE = 0
               for min in range(Nmins - 1):
                    RTE += ATE_vec[minute_idx * min: minute_idx * (min + 1)].mean()
               if Nmins > 0:
                    RTE /= Nmins
               else:
                    RTE = ATE * 60 / Nseconds

               if i == 0:
                    print("Predicting positions for ", Ntest * dt, " seconds in ", Ntest  /seq_len, " steps")
               print(f'{name_list[i]} ATE: {ATE:.3f} m')
               print(f'{name_list[i]} RTE: {RTE:.3f} m')

               ax2.plot(q_gts[:,0], q_gts[:,1], f'{marker_list[i]}', alpha=alpha_list[i], label = f'GT for {name_list[i]}', color=f'{color_list_true[i]}')
               ax2.plot(q_preds[:,0], q_preds[:,1], f'{marker_list[i]}', alpha=alpha_list[i], label = f'Pred for {name_list[i]}', color=f'{color_list_pred[i]}')

               ax2.legend()
               ax2.set_xlabel('x (m)')
               ax2.set_ylabel('y (m)')

               d = dict(ATE = ATE, RTE = RTE, Time = Nseconds)
               text = nice_string_output(d, extra_spacing=2, decimals=2)
               add_text_to_ax(0.1, 0.95, text, ax2, fontsize=13)

   
     if plot_testing:
          

          fig2, ax2 = plt.subplots()
          data = np.load('preds_lin.npz')
          lst = data.files
          y_pred_linear = data['y_pred_test']
          y_test_linear = data['y_test']

          y_pred_lstm = np.loadtxt(f'predictions_arr2.txt', delimiter=',')
      
          pred_list = [y_pred_linear, y_pred_lstm]
          pred_offset = [y_test_offset, y_test_offset]
          pred_true = [y_test_linear, y_test]

          name_list = ['Linear', 'LSTM']
          marker_list = ['-','--','--']
          alpha_list = [0.6,0.6,0.4]
          color_list_true = ['green', 'olive', 'green']
          color_list_pred = ['red', 'teal', 'red']

          for i, pred in enumerate(pred_list):

               if 0:
                    y_pred = np.array(pred)
                    y_pred = y_pred.reshape(-1, Noutput_features)
                    y_pred = y_pred + np.vstack([pred_offset[i][0,:], pred_offset[i][0,:] + np.cumsum(y_pred[:-1], axis=0)])
                    # start predictions at the initial ground truth position
                    y_pred = np.vstack([pred_offset[i][0,:], y_pred])

                    # save predictions
                    #np.savetxt(f'predictions_arr{trial_no}.txt', y_pred, delimiter=',')

                    # Include only as many test points as is divisible by batch size
                    q_gts = pred_true[i].reshape(-1, Noutput_features)
                    q_gts = q_gts + pred_offset[i]
                    q_gts = np.vstack([pred_offset[i][0,:], q_gts])
                    q_preds = y_pred
                    xs = np.arange(len(q_gts))
              
               print(pred_true[i].shape, pred_offset[i].shape)
               if i == 1:
                    # Include only as many test points as is divisible by batch size
                    q_gts = pred_true[i].reshape(-1, Noutput_features)
                    print(q_gts.shape, pred_offset[i].shape)
                    q_gts = q_gts + pred_offset[i]
                    q_gts = np.vstack([pred_offset[i][0,:], q_gts])
                    q_preds = y_pred
                    xs = np.arange(len(q_gts))
               else:
                    q_gts = pred_true[i][:,:2]
               y_pred = pred_list[i][:,:2]
               q_preds = y_pred

               dt = 1 / 200
               ATE_vec = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1))
               ATE = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1)).mean()
               Nseconds = Ntest * dt
               Nmins = int(np.floor(Nseconds / 60))
               minute_idx = int(Ntest * 60 / Nseconds / seq_len)
               RTE = 0
               for min in range(Nmins - 1):
                    RTE += ATE_vec[minute_idx * min: minute_idx * (min + 1)].mean()
               if Nmins > 0:
                    RTE /= Nmins
               else:
                    RTE = ATE * 60 / Nseconds


               print("Predicting positions for ", Ntest * dt, " seconds in ", Ntest  /seq_len, " steps")
               print(f'{name_list[i]} ATE: {ATE:.3f} m')
               print(f'{name_list[i]} RTE: {RTE:.3f} m')

               if i == 0:
                    ax2.plot(q_gts[:,0], q_gts[:,1], f'{marker_list[i]}', alpha=alpha_list[i],\
                              label = f'GT for {name_list[i]}', color=f'green')

               ax2.plot(q_preds[:,0], q_preds[:,1], lw = 1.4, alpha=alpha_list[i], label = rf'{name_list[i]} pred. (ATE={ATE:.2f})', color=f'{color_list_pred[i]}')
               ax2.tick_params(axis='x',colors='white')
               ax2.tick_params(axis='y',colors='white')
      
               ax2.xaxis.label.set_color('white')
               ax2.yaxis.label.set_color('white')
               ax2.set_title('Predictions for unseen data', color='white')
               ax2.legend()
               ax2.set_xlabel('x (m)', color='white')
               ax2.set_ylabel('y (m)', color='white')
               fig2.patch.set_alpha(0.0)

          plt.savefig('preds_lin_lstm.png', dpi=420)#, transparent=True)
              # d = dict(ATE = ATE, RTE = RTE, Time = Nseconds)
               #text = nice_string_output(d, extra_spacing=2, decimals=2)
               #add_text_to_ax(0.1, 0.95, text, ax2, fontsize=13)

   
     if do_many_tests:
          ## TEST ------------------------------------------------------

          test_folder_path = 'C:/Users/Simon Andersen/Documents/Uni/KS6/AppliedML/Project 2/unseen_subjects_test_set'


          test_dirs = os.listdir(test_folder_path)
          test_path_list = []
          # Choose which unseen subjects to test on
 
          # Collect all test paths
          for dir in test_dirs:
               test_path_list.append(test_folder_path + '/' + dir + '/' + 'data.hdf5')

          # Initialize lists for plotting
          name_list = test_dirs
          ATE_arr = np.zeros(len(test_path_list))
          RTE_arr = np.zeros(len(test_path_list))
          Ntot = 0

          # Load and predict on test data set
          for i, path in enumerate(test_path_list):
               # Load test data
               data_dict = load_data(path, verbose=False)
               y_test = data_dict[output_features[0]][:,:2]
               Npoints = y_test.shape[0]
               X_test = np.zeros([Npoints, Nfeatures])
               Ntot += Npoints
               for j, feature in enumerate(input_features):
                    if j == 0:
                         X_test = data_dict[feature]
                    else:

                         X_test = np.hstack([X_test, data_dict[feature]])


               # normalize
               X_test, _ = normalize_features(X_test, scaler=scaler)
               X_test, y_test = create_sequences(X_test, y_test, seq_length=seq_len, overlap = overlap)

               y_test_offset = y_test[:, 0, :]
               # Use only last sequence value relative to first sequence value as output
               y_test = y_test[:,-1,:] - y_test_offset

               # Store true values and offsets
               pred_true = y_test
               pred_offset = y_test_offset

               # convert to torch tensors
               X_test = torch.from_numpy(X_test).float()
               y_test = torch.from_numpy(y_test).float()

               test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
               test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

               pred = []
               with torch.no_grad():
                         for k, (inputs, labels) in enumerate(test_loader):
                              inputs, labels = inputs.to(device), labels.to(device)
                    
                              outputs = net(inputs)    
                              # copy to cpu and store prediction
                              outputs_cpu = torch.Tensor.cpu(outputs)
                              pred.append(outputs_cpu.numpy())
                              y_pred_tensor = outputs


               # CALC ATE AND RTE
               y_pred = np.array(pred)
               y_pred = y_pred.reshape(-1, Noutput_features)
               y_pred = y_pred + np.vstack([pred_offset[0], pred_offset[0] + np.cumsum(y_pred[:-1], axis=0)])
               # start predictions at the initial ground truth position
               y_pred = np.vstack([pred_offset[0], y_pred])
               # Include only as many test points as is divisible by batch size
               q_gts = pred_true.reshape(-1, Noutput_features)
               q_gts = q_gts + pred_offset
               q_gts = np.vstack([pred_offset[0], q_gts])
               q_preds = y_pred
               xs = np.arange(len(q_gts))
               Ntest = len(xs) * seq_len

               # Calc. ATE (Mean av. trajectory error) and RTE (Relative av. trajectory error)
               ATE_vec = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1))
               ATE = ATE_vec.mean()
               Nseconds = Ntest * dt
               Nmins = int(np.floor(Nseconds / 60))
               minute_idx = int(Ntest * 60 / Nseconds / seq_len)
               RTE = 0
               for min in range(Nmins - 1):
                    RTE += ATE_vec[minute_idx * min: minute_idx * (min + 1)].mean()

               RTE /= Nmins
               ATE_arr[i] = ATE
               RTE_arr[i] = RTE

               print('For dataset: ', name_list[i], ' with ', Ntest, ' test points')
               print("Predicting positions for ", Ntest * dt, " seconds in ", Ntest  /seq_len, " steps")
               print(f'{name_list[i]} ATE: {ATE:.3f} m')
               print(f'{name_list[i]} RTE: {RTE:.3f} m')

          # Mean and std of ATE
          print('Mean ATE: ', ATE_arr.mean(), ' +/- ', ATE_arr.std(ddof=1))
          print('Mean RTE: ', RTE_arr.mean(), ' +/- ', RTE_arr.std(ddof=1))
          print('Total number of test points: ', Ntot)
        

     if do_testing:
          ## TEST ------------------------------------------------------

          # Set test directories
          # Below is the path to the (to the model) unseen a000_1
          train_set_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1\\a000_1\\data.hdf5'
          # Below ist the path to the (to the model) unseen a046_2 (called seen cuz trained on other data set with same subject)
          test_seen_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\seen_subjects_test_set\\a046_2\\data.hdf5'
          test_seen_path2 = "C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\seen_subjects_test_set\\a010_2\\data.hdf5"
          # Below is the folder path to the unseen subjects
          test_folder_path = 'C:/Users/Simon Andersen/Documents/Uni/KS6/AppliedML/Project 2/unseen_subjects_test_set'

          test_path_list = [train_set_path, test_seen_path2, test_seen_path2]
          test_dirs = os.listdir(test_folder_path)
          # Choose which unseen subjects to test on
          #test_dirs = ['a006_2', 'a019_3', 'a019_3']
          test_dirs = ['a006_2', 'a053_1', 'a052_2']

          # good ones: a055_2, a053_1, a052_2, a024_3, 

          Ntest_list = [67_000, 50_000, 136_000, 73_000, 50_000, 30_000]
          Nstart = [0, 5000, 5000, 5_000, 5_000, 7000]
          # Collect all test paths
          for dir in test_dirs:
               test_path_list.append(test_folder_path + '/' + dir + '/' + 'data.hdf5')

          # Initialize lists for plotting
          pred_list = []
          pred_true = []
          pred_offset = []
          name_list = ['a000_1', 'a046_2', 'a010_2'] + test_dirs

          # Load and predict on test data set
          for i, path in enumerate(test_path_list):
               # Load test data
               data_dict = load_data(path, verbose=False)
               y_test = data_dict[output_features[0]][:,:2]
               Npoints = y_test.shape[0]
               X_test = np.zeros([Npoints, Nfeatures])
          
               for j, feature in enumerate(input_features):
                    if j == 0:
                         X_test = data_dict[feature]
                    else:

                         X_test = np.hstack([X_test, data_dict[feature]])

               X_test = X_test[Nstart[i]:Ntest_list[i], :]
               y_test = y_test[Nstart[i]:Ntest_list[i], :]

               # normalize
               X_test, _ = normalize_features(X_test, scaler=scaler)

               X_test, y_test = create_sequences(X_test, y_test, seq_length=seq_len, overlap = overlap)

          
               y_test_offset = y_test[:, 0, :]
               # Use only last sequence value relative to first sequence value as output
               y_test = y_test[:,-1,:] - y_test_offset

               # Store true values and offsets
               pred_true.append(y_test)
               pred_offset.append(y_test_offset)


               # convert to torch tensors
               X_test = torch.from_numpy(X_test).float()
               y_test = torch.from_numpy(y_test).float()

               test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
               test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

               pred = []
               with torch.no_grad():
                         for i, (inputs, labels) in enumerate(test_loader):
                              inputs, labels = inputs.to(device), labels.to(device)
                    
                              outputs = net(inputs)    
                              # copy to cpu and store prediction
                              outputs_cpu = torch.Tensor.cpu(outputs)
                              pred.append(outputs_cpu.numpy())
                              y_pred_tensor = outputs

               pred_list.append(pred)   


          fig2, ax2 = plt.subplots(nrows = 2, ncols = 3, figsize = (15,10))
          ax2 = ax2.flatten()

          # Plot predictions
          for i, pred in enumerate(pred_list):

               y_pred = np.array(pred)
               y_pred = y_pred.reshape(-1, Noutput_features)
               y_pred = y_pred + np.vstack([pred_offset[i][0,:], pred_offset[i][0,:] + np.cumsum(y_pred[:-1], axis=0)])
               # start predictions at the initial ground truth position
               y_pred = np.vstack([pred_offset[i][0,:], y_pred])
               # Include only as many test points as is divisible by batch size
               q_gts = pred_true[i].reshape(-1, Noutput_features)
               q_gts = q_gts + pred_offset[i]
               q_gts = np.vstack([pred_offset[i][0,:], q_gts])
               q_preds = y_pred
               xs = np.arange(len(q_gts))
               Ntest = len(xs) * seq_len

               # Calc. ATE (Mean av. trajectory error) and RTE (Relative av. trajectory error)
               ATE_vec = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1))
               ATE = ATE_vec.mean()
               Nseconds = Ntest * dt
               Nmins = int(np.floor(Nseconds / 60))
               minute_idx = int(Ntest * 60 / Nseconds / seq_len)
               RTE = 0
               for min in range(Nmins - 1):
                    RTE += ATE_vec[minute_idx * min: minute_idx * (min + 1)].mean()

               RTE /= Nmins

               print('For dataset: ', name_list[i], ' with ', Ntest, ' test points')
               print("Predicting positions for ", Ntest * dt, " seconds in ", Ntest  /seq_len, " steps")
               print(f'{name_list[i]} ATE: {ATE:.3f} m')
               print(f'{name_list[i]} RTE: {RTE:.3f} m')
               
               
               if i == 0:
                    gt_label = 'Ground truth'
                    pred_label = 'Predictions'
               else:
                    gt_label = None
                    pred_label = None
               ax2[i].plot(q_gts[:,0], q_gts[:,1], alpha=.7, label = gt_label, color='green')
               ax2[i].plot(q_preds[:,0], q_preds[:,1], alpha=.7, label = pred_label, color='red')

               jump = int(np.floor(len(q_gts) / 30))
               for i in range(-1):
                    x_range = np.arange(i * jump, (i + 10) * jump)
                    ax2[i].arrow(q_gts[x_range[i],0], q_gts[x_range[i],1], q_gts[x_range[i],0] + 10, q_gts[x_range[i],1] +10, shape='full', lw=0, length_includes_head=True, head_width=1)
                    ax2[i].arrow(q_preds[x_range[i],0], q_preds[x_range[i],1], q_preds[x_range[i],0] + 10, q_preds[x_range[i],1] +10, shape='full', lw=0, length_includes_head=True, head_width=1)

               ax2[i].tick_params(axis='x',colors='white')
               ax2[i].tick_params(axis='y',colors='white')
      
               ax2[i].xaxis.label.set_color('white')
               ax2[i].yaxis.label.set_color('white')
               
     
               fig2.patch.set_alpha(0.0)

          
          #  ax2[i].set_xlabel('x (m)')
          # ax2[i].set_ylabel('y (m)')
               d = dict(ATE = ATE, RTE = RTE, Time = Nseconds)
               text = nice_string_output(d, extra_spacing=2, decimals=2)
               add_text_to_ax(0.1, 0.95, text, ax2[i], fontsize=13)
     
          fig2.legend(loc = 'upper center', ncol=2, fontsize = '15')
          #fig2.suptitle('Predictions for unseen data')
          fig2.supxlabel('x (m)', fontsize = '18', color = 'white')
          fig2.supylabel('y (m)', fontsize = '18', color = 'white')

          plt.savefig(f'best{model_name}{n_trial}.png', dpi=420, bbox_inches='tight', )


     plt.show()



if __name__ == '__main__':
    main()
