# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import sys 
if '..' not in sys.path:
    sys.path.append('../')
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
#folder_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1'
folder_path = '/Users/antongolles/Documents/uni/masters/myMasters/applied_machine_learning/inertial_navigation_transformer/data/data_from_RoNIN/train_dataset_1/'


### FUNCTIONS ----------------------------------------------------------------------------------


# plot
def plot_quat(q_gts, q_preds, xs, names = ['analytical', 'NN']):
    if type(q_preds) is not list:
        q_preds = [q_preds]
    if type(q_gts) is not list:
        q_gts = [q_gts]
    if type(xs) is not list:
        xs = [xs]
    
    fig, ax = plt.subplots(len(q_preds), 4, figsize=(12, 4*len(q_preds)))
    ax = ax.reshape(-1, 4)
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



### MAIN ---------------------------------------------------------------------------------------



def main():
    ### Implement vanilla LSTM
    ## Set central parameters
     # Set how many observations to combine into one datapoint
     Ntrain = 100_000
     Ntest = 5_000
     num_datasets = 50
     # set sequence length
     seq_len = 10
     random_start = True
     # Set whether to include acceleration and magnetic data when training 
     include_acc = False
     include_lin_acc = True
     include_mag = True
     normalize=True
     overlap = 1
     # set whether to include truth data when training. 
     # ... and to include the previous test prediction as the input when testing net batch
     use_truth_input = True
     include_theta = True

     learning_rate = 0.0001
     # Set loss function
     criterion = nn.L1Loss()
     # set n_epochs
     num_epochs = 50
     # set size of hidden layers
     hidden_size = 132
     # set batch size
     batch_size = 128#seq_len * hidden_size
     # set no. of layers in LSTM
     num_layers = 6


     Ntest_fraction = Ntest / Ntrain
     # Decide which fraction of the training data should be used for validation
     Nval_fraction = .2

     

     input_features = ['synced/gyro']
     output_features = ['pose/tango_ori']
     if include_acc:
          input_features.append('synced/acce')
     if include_mag:
          input_features.append('synced/magnet')
     if include_lin_acc:
          input_features.append('synced/linacce')
     if use_truth_input:
          input_features.append(output_features[0])

  
     


     params = {'N_train': Ntrain, 
               'N_test': Ntest,
               'seq_len': seq_len, 
               'input': input_features,
               'output': output_features, 
               'normalize': normalize,
               'verbose': False, 
               'num_datasets':num_datasets,
               'random_start':random_start,
               'overlap': overlap,
               'include_theta': include_theta,}
     X_train, y_train, X_test, y_test, col_locations = load_split_data(folder_path=folder_path, **params)
     print(col_locations)
     Noutput_features = y_train.shape[2]
     # Discard last point of X and first point of y
     print('X', X_train.shape, 'y', y_train.shape)
     X_train[:, 1:, col_locations['input']['pose/tango_ori'][0]:col_locations['input']['pose/tango_ori'][1]] = 0
     X_test[:, 1:, col_locations['input']['pose/tango_ori'][0]:col_locations['input']['pose/tango_ori'][1]] = 0
     
     #reshape y
     y_train = y_train.reshape(y_train.shape[0], - 1) #y_train[:,-1,:]
     #y_test = y_test.reshape(y_test.shape[0], - 1) #y_test[:,-1,:]
     
     Nfeatures = X_train.shape[-1]

     class LSTM(nn.Module):
          def __init__(self, input_size, hidden_size=7, num_layers = 1, output_size = 4, seq_len = 1,):
               super(LSTM, self).__init__()
               self.num_layers = num_layers
               self.hidden_size = hidden_size
               self.relu = nn.ReLU()

               # pre
               self.input_linear = nn.Linear(input_size, input_size*2)
               self.input_linear2 = nn.Linear(input_size*2, input_size*2)
               self.input_linear3 = nn.Linear(input_size*2, input_size)
               
               # main
               self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, \
                                   num_layers = num_layers, batch_first=True)
               self.linear_main = nn.Linear(hidden_size, 16)
               
               # post
               self.linear_post_1 = nn.Linear(36, hidden_size*2)

               self.linear_post_2 = nn.Linear(hidden_size*2, hidden_size*2)
               self.linear_post_3 = nn.Linear(hidden_size*2, hidden_size)
               self.linear_post_4 = nn.Linear(hidden_size, output_size)
               


          def forward(self, x):
               # read input
               gyro = x[:,:,col_locations['input']['synced/gyro'][0]:col_locations['input']['synced/gyro'][1]]
               mag = x[:,:,col_locations['input']['synced/magnet'][0]:col_locations['input']['synced/magnet'][1]]
               linacce = x[:,:,col_locations['input']['synced/linacce'][0]:col_locations['input']['synced/linacce'][1]]
               X_main = torch.cat((gyro, mag, linacce), dim=2)
               ori_0 = x[:,0,col_locations['input']['pose/tango_ori'][0]:col_locations['input']['pose/tango_ori'][1]]
               theta = x[:,0,col_locations['input']['theta'][0]:col_locations['input']['theta'][1]]
               
               # pre
               X_main = self.input_linear(X_main)
               X_main = self.input_linear2(X_main)
               X_main = self.input_linear3(X_main)
               
               # main
               X_main_lstm, _ = self.lstm(X_main)  # Run LSTM and store the hidden layer outputs
               X_main_lstm = X_main_lstm[:, -1, :]  # take the last hidden layer
               X_main_lstm = self.linear_main(X_main_lstm) # a normal dense layer
               # concatenate gyro_lstm and ori_0
               #print(gyro_lstm.shape, ori_0.shape)
               concatted = torch.cat((X_main_lstm, ori_0, theta), dim=1)

               # post
               concatted = self.linear_post_1(concatted)
               concatted = self.relu(concatted)
               concatted = self.linear_post_2(concatted)
               concatted = self.relu(concatted)
               concatted = self.linear_post_3(concatted)
               concatted = self.relu(concatted)
               concatted = self.linear_post_4(concatted)
               return concatted
          
          # set all weight to zero
          def zero_weights(self):
               for m in self.modules():
                    if isinstance(m, nn.Linear):
                         m.weight.data.zero_()
                         m.bias.data.zero_()
                    elif isinstance(m, nn.LSTM):
                         for name, param in m.named_parameters():
                              if 'weight_ih' in name:
                                   torch.nn.init.xavier_uniform_(param.data)
                              elif 'weight_hh' in name:
                                   torch.nn.init.orthogonal_(param.data)
                              elif 'bias' in name:
                                   param.data.zero_()


     # setting device on GPU if available, else CPU
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print('Using device:', device)

     #Additional Info when using cuda
     if device.type == 'cuda':
          print(torch.cuda.get_device_name(0))
          print('Memory Usage:')
          print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
          print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

     # initialize model
     output_size = Noutput_features * seq_len


     net = LSTM(input_size=9, output_size = output_size,hidden_size=hidden_size,\
                num_layers=num_layers,  seq_len=seq_len)
     # mount model to device
     net.to(device)
     # number of params
     num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
     print('Number of parameters: %d' % num_params)
     
     ## Prepare and split data
     # train test split
     # Now split train data into train and val. data
     X_train_arr, X_val_arr, y_train_arr, y_val_arr = train_test_split(X_train, y_train, test_size=Nval_fraction, shuffle=False)

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
                         if inputs.shape[0] != y_pred_tensor.shape[0]:
                              break
                         else:
                              inputs[:,:,col_locations['input']['pose/tango_ori'][0]:col_locations['input']['pose/tango_ori'][1]] = 0
                              #inputs[:,0,-Noutput_features:] = y_pred_tensor.reshape(y_pred_tensor.shape[0], seq_len, Noutput_features)[:,0,:]
                              inputs[:,0,col_locations['input']['pose/tango_ori'][0]:col_locations['input']['pose/tango_ori'][1]] = y_pred_tensor.reshape(y_pred_tensor.shape[0], seq_len, Noutput_features)[0, -1]

                    #print(inputs)
                    outputs = net(inputs)
                    #print(outputs)
                    # copy to cpu and store prediction
                    outputs_cpu = torch.Tensor.cpu(outputs)
                    y_pred.append(outputs_cpu.numpy())
                    y_pred_tensor = outputs.reshape(outputs.shape[0], seq_len, Noutput_features)
                    #print(y_pred_tensor.shape)
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
     print('y_pred', y_pred.shape)
     print('y_test_arr', y_test.shape)
     # reshape from (Nbatches, BatchSize, seq_len * Noutput features) to (Npoints = Nbatches * Batchsize, seq_len * Noutput_features)
     y_pred = y_pred.reshape(-1, Noutput_features)
     # only plot the last prediction for each sequence
     #y_pred = y_pred[:, -Noutput_features:]

     # Include only as many test points as is divisible by batch size
     q_gts = y_test.reshape(-1, Noutput_features)
     # Only plot last true value for each sequence
     q_gts = q_gts[:len(y_pred)]

     q_preds = y_pred
     xs = np.arange(len(q_gts))

     fig, ax = plt.subplots(nrows=2,ncols=2)
     ax = ax.flatten()
     for i, axx in enumerate(ax):
          axx.plot(xs, q_gts[:,i], label = 'GT', color='green',)
          axx.plot(xs, q_preds[:,i], label = 'Pred', color='red',)
          axx.legend()
  
     plt.show()


if __name__ == '__main__':
    main()
