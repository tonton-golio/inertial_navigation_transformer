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

### FUNCTIONS ----------------------------------------------------------------------------------


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




### MAIN ---------------------------------------------------------------------------------------

def main():
    ### Implement vanilla LSTM
    ## Set central parameters
     # Set how many observations to combine into one datapoint
     Ntrain = 50_000
     Ntest = 1500
     num_datasets = 10
     random_start = True
     overlap = 1
     # set sequence length
     seq_len = 5
     # Set whether to include acceleration and magnetic data when training 
     include_acc = True
     include_lin_acc = False
     include_mag = True
     normalize = True
     # set whether to include truth data when training. 
     # ... and to include the previous test prediction as the input when testing net batch
     use_truth_input = True

     learning_rate = 0.0003
     # Set loss function
     criterion = nn.L1Loss()
     # set n_epochs
     num_epochs = 100
     # set size of hidden layers
     hidden_size = 80
     # set batch size
     batch_size = 128
     # set no. of layers in LSTM
     num_layers = 4
     # set the dropout layer strength
     dropout_layer = 0.1

     Ntest_fraction = Ntest / Ntrain
     # Decide which fraction of the training data should be used for validation
     Nval_fraction = .2

     
     output_features = ['pose/tango_ori']
     input_features = ['synced/gyro']
     if include_acc:
          input_features.append('synced/acce')
     if include_lin_acc:
          input_features.append('synced/linacce')
     if include_mag:
          input_features.append('synced/magnet')
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
     X_train, y_train, X_test, y_test = load_split_data(folder_path=folder_path, **params)
     
     
     # Set all but the first true value to 0 in each sequence
     Noutput_features = y_train.shape[-1]
     X_train[:, 1:, -Noutput_features:] = 0
     X_test[:, 1:, -Noutput_features:] = 0
     
     # Include only the last true value in each sequence
     y_train = y_train[:,-1,:]#.reshape(y_train.shape[0], - 1)
     y_test = y_test[:,-1,:]#.reshape(y_test.shape[0], - 1)
     
     print('Xtrain', X_train.shape, 'ytrain', y_train.shape)
     print('Xtest', X_test.shape, 'ytest', y_test.shape)

     output_size = y_train.shape[-1]
     Noutput_features = output_size
     Nfeatures = X_train.shape[-1]


     ## Prepare and split train data into train and val. data
     X_train_arr, X_val_arr, y_train_arr, y_val_arr = \
          train_test_split(X_train, y_train, \
                                                       test_size=Nval_fraction, shuffle=False)
    

     print(X_train_arr.shape, X_val_arr.shape, y_train_arr.shape, y_val_arr.shape)

     # convert to torch tensors
     X_train = torch.from_numpy(X_train_arr).float()
     y_train = torch.from_numpy(y_train_arr).float()
     X_val = torch.from_numpy(X_val_arr).float()
     y_val = torch.from_numpy(y_val_arr).float()
     X_test = torch.from_numpy(X_test).float()
     y_test = torch.from_numpy(y_test).float()

     # dataset
     train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
     val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
     test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

     # dataloader
     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
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

     # Train
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
          print('Epoch %d, loss: %.5f' % (epoch+1, running_loss/n_minibatches))

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
               print('Test loss: %.5f' % (running_loss/n_minibatches))

     t_end = time.time()
     print("Training time: ", t_end - t_start)

     
     y_pred = []
     if use_truth_input:
          with torch.no_grad():
               for i, (inputs, labels) in enumerate(test_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    if i > 0:
                               # copy last prediction to input
                              inputs[:,0,-Noutput_features:] = y_pred_tensor[0]

                    outputs = net(inputs)
                    # copy to cpu and store prediction
                    outputs_cpu = torch.Tensor.cpu(outputs)
                    y_pred.append(outputs_cpu.numpy())
                    y_pred_tensor = outputs
     else:
          with torch.no_grad():
               for i, (inputs, labels) in enumerate(test_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    if inputs.shape[0] < batch_size:
                         break
                    outputs = net(inputs)
                    # copy to cpu and store prediction
                    outputs_cpu = torch.Tensor.cpu(outputs)
                    y_pred.append(outputs_cpu.numpy())
                    y_pred_tensor = outputs
     


     y_pred = np.array(y_pred)
     y_pred = y_pred.reshape(-1, Noutput_features)
 
     # Include only as many test points as is divisible by batch size
     q_gts = y_test.reshape(-1, Noutput_features)
     q_gts = q_gts[:len(y_pred)]

     q_preds = y_pred
     xs = np.arange(len(q_gts))

     nrows = 1 if Noutput_features == 3 else 2
     ncols = 3 if Noutput_features == 3 else 2
     fig, ax = plt.subplots(nrows=nrows,ncols=ncols)
     ax = ax.flatten()

     for i, axx in enumerate(ax):
          try:
               axx.plot(xs, q_gts[:,i], label = 'GT', color='green',)
               axx.plot(xs, q_preds[:,i], label = 'Pred', color='red')
               axx.legend()
          except:
               break
  
     plt.show()

if __name__ == '__main__':
    main()
