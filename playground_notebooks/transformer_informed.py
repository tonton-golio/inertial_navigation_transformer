import sys 
if '..' not in sys.path:
    sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

# import rotations
from scipy.spatial.transform import Rotation as R

import seaborn as sns
from matplotlib import rcParams

sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
rcParams.update(d)

# Define constant variables
FOLDER_PATH = '/Users/antongolles/Documents/uni/masters/myMasters/applied_machine_learning/inertial_navigation_transformer/data/data_from_RoNIN/train_dataset_1/'
N_TRAIN = 1600_000
N_VAL_FRACTION = .2
N_TEST = 67_000
NUM_DATASETS = 49
SEQ_LEN = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.000_003
N_EPOCHS = 300


# Set input and output features
INPUT_FEATURES = ['synced/gyro', 'synced/acce', 'synced/magnet', 'pose/tango_ori']
OUTPUT_FEATURES = ['pose/tango_pos']

params_load_data = {'N_train': N_TRAIN, 
        'N_test': N_TEST,
        'seq_len': SEQ_LEN, 
        'input': INPUT_FEATURES,
        'output': OUTPUT_FEATURES, 
        'normalize': True,
        'verbose': False, 
        'num_datasets': NUM_DATASETS,
        'random_start': False,
        'overlap': 1,
        'include_theta': False}
print('Loading data...')
X_train, y_train, X_test, y_test, col_locations = load_split_data(folder_path=FOLDER_PATH, **params_load_data)
print(col_locations)

# get validation set

def preprocess_data(X, y, col_locations, test=False):
    
    print('Preprocessing data...')
    # for y, subtract the first value from all values
    print(y.shape)
    if test == False:
        y -= y[:, :1, :]
    

    # for X, use gyro to calculate theta and get the predicted orientations
    # but first, get initial orientation
    acc_cols, mag_cols, gyro_cols = col_locations['input']['synced/acce'], col_locations['input']['synced/magnet'], col_locations['input']['synced/gyro']

    #############   Getting world frame
    def get_body2world_rot(m0,a0):
        #print(m0, a0)
        EAST = np.cross(m0,a0)
        EAST/=np.linalg.norm(EAST)
        DOWN = a0                                          # maybe a minus sign is needed
        DOWN/=np.linalg.norm(DOWN)
        NORTH = np.cross(EAST,DOWN)
        NORTH/=np.linalg.norm(NORTH)

        # body frame to world frame
        R = np.array([NORTH, EAST, DOWN]).T
        return R

    def rotation_matrix_2_quaternion(R):                 # they may not be in the right order
        # R is a 3x3 rotation matrix
        # q is a 4x1 quaternion
        q = np.zeros((4,1))
        q[0] = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])/2
        q[1] = (R[2,1] - R[1,2])/(4*q[0])
        q[2] = (R[0,2] - R[2,0])/(4*q[0])
        q[3] = (R[1,0] - R[0,1])/(4*q[0])
        return q

    def get_orientations(X):
        initial_orientation = [rotation_matrix_2_quaternion(get_body2world_rot(
                                X[seq, 0, acc_cols[0]:acc_cols[1]], 
                                X[seq, 0, mag_cols[0]:mag_cols[1]]))

                                for seq in range(X.shape[0])]
        initial_orientation = np.array(initial_orientation).reshape(-1, 1, 4)
        #print(initial_orientation)
        # get predicted orientations
        orientations = np.zeros((X.shape[0], X.shape[1], 4))
        orientations[:, 0, :] = initial_orientation.reshape(-1, 4)
        for seq in range(X.shape[0]):
            for i in range(1, X.shape[1]):

                #q_pred[i] = Theta(w[i]*factor, dt=dt)@q_pred[i-1]
                orientations[seq, i] = Theta(X[seq, i, gyro_cols[0]:gyro_cols[1]], dt=1/200)@orientations[seq, i-1]
                orientations[seq, i] /= np.linalg.norm(orientations[seq, i])
                #print(orientations[seq, i])
        return orientations
    
    if 'synced/tango_ori' in col_locations['input']:
        orientations = X[:, :, col_locations['input']['synced/tango_ori'][0]:
                                col_locations['input']['synced/tango_ori'][1]]
        
    else:
        orientations = get_orientations(X)
    #print(orientations)
    
    # subtract first acc value from all acc values
    X[:, :, acc_cols[0]:acc_cols[1]] -= X[:, :1, acc_cols[0]:acc_cols[1]]

    # rotate all acc values to world frame
    for seq in range(X.shape[0]):
        for i in range(X.shape[1]):
            X[seq, i, acc_cols[0]:acc_cols[1]] = R.from_quat(orientations[seq, i]).apply(X[seq, i, acc_cols[0]:acc_cols[1]])

    # velocity is the integral of acceleration
    v = np.cumsum(X[:, :, acc_cols[0]:acc_cols[1]], axis=1)*1/200
    print('v',v.shape)
    
    # position is the integral of velocity plus half of the integral of acceleration squared
    p = np.cumsum(v, axis=1)*1/200 + np.cumsum(X[:, :, acc_cols[0]:acc_cols[1]]**2, axis=1)*1/200**2/2

    # add velocity and position to X

    X = np.concatenate((X, orientations, v, p), axis=2)
    print(X.shape)
    

    # update col_locations
    col_locations['input']['calced/orientations'] = [X.shape[2]-10, X.shape[2]-6]
    col_locations['input']['calced/velocity'] = [X.shape[2]-6, X.shape[2]-3]
    col_locations['input']['calced/position'] = [X.shape[2]-3, X.shape[2]]
    
    X = X.reshape(X.shape[0], -1)
    y = y[:,-1,:] 
    print('final X shape: ', X.shape, '\nfinal y shape: ', y.shape)
    return X, y, col_locations
# preprocess data

X_train,X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=N_VAL_FRACTION, random_state=42)

X_train, y_train, col_locations = preprocess_data(X_train, y_train, col_locations)
X_val, y_val, _ = preprocess_data(X_val, y_val, col_locations)
X_test, y_test, _ = preprocess_data(X_test, y_test, col_locations, test=True)

print('X_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape, 
      '\nX_val shape: ', X_val.shape, '\ny_val shape: ', y_val.shape,
      '\nX_test shape: ', X_test.shape, '\ny_test shape: ', y_test.shape)
dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)

dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# model
# set up a tansformer model
timestamp_ = datetime.now().strftime("%Y%m%d-%H%M%S")
class Net(nn.Module):
    def __init__(self, input_size=23, hidden_size=1000, output_size=3, trans_layers=5, dropout=0.001):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_size*SEQ_LEN, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, input_size)
        )

        self.transformer = nn.Transformer(d_model=input_size, nhead=1, num_encoder_layers=trans_layers, num_decoder_layers=trans_layers, dim_feedforward=200, dropout=dropout, batch_first=True)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size, 
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )   

        # set all weights to zero
        # set all the weights to be the same
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # #set all the weights to be 0
        # for p in self.parameters():
        #     p.data.fill_(0)
    def forward(self, x):
        x = self.fc_in(x)
        # check for nan
        #print('first step nan check',np.isnan(x.detach().numpy()).any() == True)
    
        #x = self.transformer(x, x)
        #x, _ = self.rnn(x)
        # select last hidden state
        
        #print('second step nan check',np.isnan(x.detach().numpy()).any() == True)
        x = self.fc_out(x)
        #print('third step nan check',np.isnan(x.detach().numpy()).any() == True)
        # select last time step
        #x = x[:, -1:, :]
        return x
    
model = Net()
model = model.to('cpu')
#print(model)

# print model param count
print('model param count: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# load weights
# weights_path = 'model_informed_linear_20230611-214601.pt'
# model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))


# training loop
# set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

losses_train = []
losses_val = []
for epoch in range(N_EPOCHS):
    try:
        # training
        model.train()
        running_loss = 0.0
        for i, (X_batch, y_batch) in enumerate(dataloader_train):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            #print(y_pred, y_batch)
            loss = criterion(y_pred
                            , y_batch
                            )
            #print(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        losses_train.append(running_loss/len(dataloader_train))
        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(dataloader_val):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                running_val_loss += loss.item()

        print('Epoch: ', epoch,'\\tTraining loss: ', running_loss/len(dataloader_train), 'Validation loss: ', running_val_loss/len(dataloader_val))
        losses_val.append(running_val_loss/len(dataloader_val))
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        break
# save model
torch.save(model.state_dict(), f'model_informed_linear_{timestamp_}.pt')                

# plot predictions
model.eval()
preds_test = []
preds_train = []
with torch.no_grad():
    for i in range(len(X_test)):
        y_pred = model(torch.from_numpy(X_test[i]).float())
        preds_test.append(y_pred.numpy())

    for i in range(len(X_train)):
        y_pred = model(torch.from_numpy(X_train[i]).float())
        preds_train.append(y_pred.numpy())


# combine predictions
y_pred_test = np.array(preds_test)
y_calced_test = X_test.reshape(-1, SEQ_LEN, 19+4)[:,-1,col_locations['input']['calced/position'][0]: 
                                  col_locations['input']['calced/position'][1]]
y_test = np.array(y_test)
y_test -= y_test[0]

y_pred_train = np.array(preds_train)
y_calced_train = X_train.reshape(-1, SEQ_LEN, 19+4)[:,-1,col_locations['input']['calced/position'][0]:
                                    col_locations['input']['calced/position'][1]]
y_train = np.array(y_train)

# use cumsum
y_pred_test = np.cumsum(y_pred_test, axis=0)
y_calced_test = np.cumsum(y_calced_test, axis=0)
y_pred_train = np.cumsum(y_pred_train, axis=0)
y_calced_train = np.cumsum(y_calced_train, axis=0)
y_train = np.cumsum(y_train, axis=0)

# save predictions
d = {
    'y_pred_test': y_pred_test,
    'y_test' :y_test,
}
np.savez(f'predictions_informed_linear_{timestamp_}.npz', **d)


print(y_pred_test.shape, y_calced_test.shape, y_test.shape)
print(y_pred_train.shape, y_calced_train.shape, y_train.shape)
# fig
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=420)
# plot losses
ax[0].set_title('Losses', color='white')
ax[0].plot(losses_train, label='train')
ax[0].plot(losses_val, label='val')
ax[0].legend()
ax[0].grid(True)  # Add gridlines
ax[0].xaxis.label.set_color('white')
ax[0].yaxis.label.set_color('white')
ax[0].set_xlabel('Epoch', color='white')
ax[0].set_ylabel('Loss', color='white')
ax[0].tick_params(axis='x',colors='white')
ax[0].tick_params(axis='y',colors='white')

# plot predictions
ax[1].set_title('Predictions in XY-plane (test data)', color='white')

ax[1].plot(y_pred_test[:, 0], y_pred_test[:,1], label='pred', alpha=0.5)
ax[1].plot(y_test[:, 0], y_test[:,1], label='gt', alpha=0.5)
ax[1].plot(y_calced_test[:, 0], y_calced_test[:,1], label='calced', alpha=0.5)
ax[1].legend()


ax[1].grid(True)  # Add gridlines
ax[1].xaxis.label.set_color('white')
ax[1].yaxis.label.set_color('white')
ax[1].set_xlabel('X', color='white')
ax[1].set_ylabel('Y', color='white')
ax[1].tick_params(axis='x',colors='white')
ax[1].tick_params(axis='y',colors='white')




# # plot predictions
# ax[2].set_title('Predictions in XY-plane (train data)')
# ax[2].plot(y_pred_train[:, 0], y_pred_train[:,1], label='pred', alpha=0.5)
# ax[2].plot(y_train[:, 0], y_train[:,1], label='gt', alpha=0.5)
# ax[2].plot(y_calced_train[:, 0], y_calced_train[:,1], label='calced', alpha=0.5)
# ax[2].legend()

fig.patch.set_alpha(0.0)
plt.tight_layout()



plt.savefig(f'../new_figures/predictions_informed_transformer_{timestamp_}.png')
plt.close()