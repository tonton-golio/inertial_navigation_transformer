# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler

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


sys.path.append('inertial_navigation_transformer')
from utils import load_much_data, load_split_data

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
folder_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1'
#folder_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_full'
### FUNCTIONS ----------------------------------------------------------------------------------
test_set_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\unseen_subjects_test_set'

# plotNoutput
def plot_quat(q_gts, q_preds, xs, names = ['analytical', 'NN'], Noutput_features = 4):
    if type(q_preds) is not list:
        q_preds = [q_preds]
    if type(q_gts) is not list:
        q_gts = [q_gts]
    if type(xs) is not list:
        xs = [xs]
    
    fig, ax = plt.subplots(len(q_preds), Noutput, figsize=(12, Noutput*len(q_preds)))
    ax = ax.reshape(-1, Noutput)
    idx = 0
    for q_pred, q_gt,x, name in zip(q_preds, q_gts,xs, names):
        print('q_pred', q_pred.shape, 'q_gt', q_gt.shape, 'x', x.shape)
        ax[idx, 0].plot(x, q_gt[:,0],)
        ax[idx, 0].plot(x, q_pred[:,0],'.')
        ax[idx, 1].plot(x, q_gt[:,1],)
        ax[idx, 1].plot(x, q_pred[:,1],)
        ax[idx, 2].plot(x, q_gt[:,2],)
        ax[idx, 2].plot(x, q_pred[:,2],)
        ax[idx, 3].plot(x, q_gt[:,3],label='gt')
        ax[idx, 3].plot(x, q_pred[:,3],label=f'pred')
        
        ax[idx, 0].set_title(name)
        idx += 1
    ax[0, 3].legend()
    fig.suptitle('Quaternion Update test')
    plt.tight_layout()

class Net(nn.Module):
          def __init__(self, input_size, hidden_size=7, num_layers = 1,\
                        output_size = 4, seq_len = 1, dropout = 0):
               super().__init__()
               self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, \
                                   num_layers = num_layers, batch_first=True, dropout = dropout)
               #self.relu = nn.ReLU()
               self.linear1 = nn.Linear(hidden_size, 5 * hidden_size)
               self.linear2 = nn.Linear(5 * hidden_size, output_size)
                    
          def forward(self, x):
               x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
               x = x[:, -1, :]  # take the last hidden layer
               #x = self.relu(x) # a normal dense layer
               x = self.linear1(x)
               x = self.linear2(x)
               return x

## Doesn't work; becomes nan with backpropagation
class FourVectorLoss(nn.Module):
    def __init__(self):
        super(FourVectorLoss, self).__init__()

    def forward(self, q1, q2):
        batch_size = q1.shape[0]
        loss = 0

        Q1 = torch.Tensor.cpu(q1)
        Q2 = torch.Tensor.cpu(q2)

        dot_prod_arr = torch.sum(Q1 * Q2, dim = 1)
        norm_prod_arr = torch.norm(Q1, dim = 1) * torch.norm(Q2, dim = 1)
        norm_prod_arr[norm_prod_arr == 0] = 1e-8  # Avoid division by zero
        cos_similarity = torch.abs(dot_prod_arr / norm_prod_arr)
        loss = torch.arccos(cos_similarity).sum()

        return loss / batch_size


     

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
    ## Set central parameters
    
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
     Ntrain = 1_200_000  #3_000_000 + 1_500_000
     Ntest = 65_000 
     num_datasets = int(Ntrain / 60_000)
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
     
     learning_rate = 0.0001
     # Set loss function
     criterion = nn.MSELoss()
     # set n_epochs

     overlap = 1
     # set sequence length
     seq_len = 30
     dt = 1 / 200


     num_epochs = 15   # set size of hidden layers
     hidden_size = 100
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
               'overlap': overlap}
     X_train, y_train, X_test, y_test = load_split_data(folder_path=folder_path, **params)[:-1]
     
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


     net = Net(input_size=Nfeatures, output_size = output_size,hidden_size=hidden_size,\
                num_layers=num_layers,  seq_len=seq_len, dropout=dropout_layer)
     # mount model to device
     net.to(device)
     # number of params
     num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
     print('Number of parameters: %d' % num_params)
     
     
     # optimizer
     optimizer = optim.Adam(net.parameters(), lr=learning_rate)
     t_start = time.time()

     # TRAIN -----------------------------------------------------------
     for epoch in range(num_epochs):
          running_loss = 0.0
          n_minibatches = 0
          for inputs, labels in train_loader:
               inputs, labels = inputs.to(device), labels.to(device)
               optimizer.zero_grad() 
               outputs = net(inputs)
               
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()
               running_loss += loss.item()
               n_minibatches += 1
          print('Epoch %d, loss: %.5f' % (epoch+1, running_loss/n_minibatches/seq_len))

          # test
          with torch.no_grad():
               running_loss = 0.0
               n_minibatches = 0
               for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                   
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    n_minibatches += 1
               print('Test loss: %.5f' % (running_loss/n_minibatches/seq_len))

     t_end = time.time()
     print("Training time: ", t_end - t_start)


     ## TEST ------------------------------------------------------

     # Get test directories
     test_dirs = os.listdir(test_set_path)
     # Set how many dataset to test on
     Ntest = 4



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
          # Include only as many test points as is divisible by batch size
          q_gts = pred_true[i].reshape(-1, Noutput_features)
          q_gts = q_gts + pred_offset[i]
          q_gts = np.vstack([pred_offset[i][0,:], q_gts])
          q_preds = y_pred
          xs = np.arange(len(q_gts))

          ATE = np.sqrt(((q_gts - y_pred) ** 2).sum(axis=1)).mean()
          RTE = ATE * (60 * (1/dt) / seq_len) / (Ntest / seq_len) if Ntest / seq_len <= 60 else ATE

          if i == 0:
               print("Predicting positions for ", Ntest * dt, " seconds in ", Ntest  /seq_len, " steps")
          print(f'{name_list[i]} ATE: {ATE:.3f} m')
          print(f'{name_list[i]} RTE: {RTE:.3f} m')

          if pred_position_in_2D:
               
               ax2.plot(q_gts[:,0], q_gts[:,1], f'{marker_list[i]}', alpha=alpha_list[i], label = f'GT for {name_list[i]}', color=f'{color_list_true[i]}')
               ax2.plot(q_preds[:,0], q_preds[:,1], f'{marker_list[i]}', alpha=alpha_list[i], label = f'Pred for {name_list[i]}', color=f'{color_list_pred[i]}')

               ax2.legend()
               ax2.set_xlabel('x (m)')
               ax2.set_ylabel('y (m)')



     plt.show()



if __name__ == '__main__':
    main()
