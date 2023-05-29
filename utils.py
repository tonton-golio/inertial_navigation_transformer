import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re


def get_all_datasets(hdf_file):
    datasets = {}

    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj[:]

    hdf_file.visititems(collect_datasets)
    return datasets

def nice_dict_contents(data_dict, print_keys=False):
    outer_keys = {}
    for key in data_dict.keys():
        num_levels = len(key.split('/'))
        outer = key.split('/')[0]
        if num_levels == 2:
            inner = key.split('/')[1]
            if outer not in outer_keys.keys():
                outer_keys[outer] = []
            outer_keys[outer].append(inner)
        else:
            middle = key.split('/')[1]
            inner = key.split('/')[2]
            if outer not in outer_keys.keys():
                outer_keys[outer] = {}
            if middle not in outer_keys[outer].keys():
                outer_keys[outer][middle] = []
            outer_keys[outer][middle].append(inner)


    if print_keys:
        print('CONTENTS OF HDF5 FILE:')
        for key, v in outer_keys.items():
            print(key)
            if isinstance(v, list):
                #for i in v:
                print('\t', ', '.join(v))
            else:
                for k, v in v.items():
                    print('\t', k)
                    #for i in v:
                    print('\t\t', ', '.join(v))

def load_data(file_path='/Users/antongolles/Documents/work/Rokoko/velocity_est/data/data_from_RoNIN/train_dataset_1/a000_1/data.hdf5', verbose=False):
    with h5py.File(file_path, 'r') as hdf_file:
        data_dict = get_all_datasets(hdf_file)
    
    if verbose:
        nice_dict_contents(data_dict, print_keys=True)

    return data_dict


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_length + 1, seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def load_much_data(Ntrain, Nval, folder_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1', \
                   columns = ['pose/tango_ori', 'pose/tango_pos', 'synced/gyro', 'synced/acce', 'synced/magnet']):
    """
    Ntrain: No. of training data points
    Nval: No. of val. data points
    folder_path: Must lead to a folder with subfolders, with each subfolder containing af hdf5 file
    """
    Nloaded_points = 0
    # Initialize data_dict
    data_dict = {key: None for key in columns}

    # Sort the subfolders based on the custom sorting key
    #sorted_folders = sorted(os.listdir(folder_path), key=folder_sort_key)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Nloaded_points > Ntrain + Nval:
                continue
            if file.endswith(".hdf5"):
                file_path = os.path.join(root, file)
                print(f"Reading file: {file_path}")
                print(file_path)
                # Read the HDF5 file
                with h5py.File(file_path, "r") as hdf_file:
                    # Access and process the contents of the HDF5 file
                    # Example: Print all the dataset names in the file

                    new_data_dict = load_data(file_path)
                    Nloaded_points += new_data_dict[columns[0]].shape[0]
                # add values to data_dict
                for key in columns:
                    if data_dict[key] is None:
                        data_dict[key] = new_data_dict[key]
                    else:
                        data_dict[key] = np.vstack([data_dict[key], new_data_dict[key]])
    ## Make sure we have Ntrain+Nval entries
    for key in columns:
        data_dict[key] = data_dict[key][:Ntrain + Nval]
    return data_dict

def load_split_data(params, folder_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1',):
    """
    params must have the following arguments
    params ={'Ntrain':, 'Nval':, 'seq_len': 'input': , 'output':}, where
    seq_len: sequence length > 1
    input: list of input features. 
            For example: ['pose/tango_ori', 'pose/tango_pos', 'synced/gyro']
    output: list of output features. 
            For example: ['pose/tango_ori']

    returns: An [Nrows/seg_len]x[seq_len]x[Nfeatures] array  X, where
             Nfeatures is the total number of feature dimensions in the input list. 
             The input features in X are in the same order as the input argument in params
             An array y with N/seq_len rows and Noutputfeatures columns, where Noutputfeatures is the
             no. of feature dimensions of the output parameter.
    """
    Ntrain, Nval = params['Ntrain'], params['Nval']
    seq_len = params['seq_len']
    input_features = params['input']
    output = params['output']
    columns = input_features + [output]
 
    data_dict = load_much_data(Ntrain, Nval, folder_path = folder_path, columns=columns)

    # Construct x and y
    for i, key in enumerate(input_features):
        if i == 0:
            X = data_dict[key]
        else:
            X = np.hstack([X, data_dict[key]])    
    y = data_dict[output]
    ## Reshape x and y
    print(X.shape, y.shape)
    X_reshaped, y_reshaped = create_sequences(X, y, seq_length=seq_len)

    return X_reshaped, y_reshaped



##

# we need a cool dataset class, that can be used for training
# we should take a batch of data like 1000 timesteps, but perhaps they should not all be this length
# we can fill zeros at the end. 


#####################

# Get quat update from gyro: https://sci-hub.hkvisa.net/10.3390/s111009182
def is_column_vector(w):
    try:
        w = w.reshape(3,1)
        return w
    except:
        print('w must be a column vector')
        return None
    
def Omega(w=np.array([1,1,2]).reshape(3,1)):
    # given a 3x1 vector w, return the 4x4 matrix Omega(w)
    # w is the angular velocity
    # Omega(w) is the matrix such that ...
    w = is_column_vector(w)
    w_cross = np.array([[0, -w[2,0], w[1,0]],
                        [w[2,0], 0, -w[0,0]],
                        [-w[1,0], w[0,0], 0]])
    top = np.hstack((-w_cross, w))
    #print('top',top)
    bottom = np.hstack((-w.T, np.zeros((1,1))))
    #print('bottom',bottom)
    #print()
    return np.vstack((top, bottom))

def u_b(w=np.array([1,1,2]).reshape(3,1), dt=.1):
    # given a 3x1 vector w, return the 4x4 matrix u_b(w)
    # w is the angular velocity
    # u_b(w) is the integration of w over the time interval dt
    w = is_column_vector(w)
    return  w*dt

# matrix norm of a 4x4 matrix
def matrix_norm(M):
    return np.sqrt(np.trace(M.T@M))

def vec_norm(v):
    return np.sqrt(v.T@v)

def Theta(w=np.array([1,1,2]).reshape(3,1), dt=.1):
    # given a 3x1 vector w, return the 4x4 matrix Theta(w)
    # w is the angular velocity
    # Theta(w) is the matrix such that ...
    w = is_column_vector(w)
    W = Omega(w)
    u = u_b(w, dt)
    W_norm = matrix_norm(W)
    u_b_norm = vec_norm(u)
    return np.cos(u_b_norm/2)*np.eye(4) + np.sin(u_b_norm/2)/(u_b_norm/2)*W


# w = data_dict['synced/gyro']
# # 100 time steps
# q_gt = data_dict['pose/tango_ori'][:100]
# dt = data_dict['synced/time'][1] - data_dict['synced/time'][0]
# q_pred = np.zeros((100,4))
# q_pred[0] = q_gt[0]
# factor= .0025                                # has been set arbitrarily
# for i in range(1,100):
#     q_pred[i] = Theta(w[i]*factor, dt=dt)@q_pred[i-1]


#############   Getting world frame
def get_body2world_rot(m0,a0):
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

def quat_2_rotation_matrix(q):
    # q is a 4x1 quaternion
    # R is a 3x3 rotation matrix
    R = np.zeros((3,3))
    R[0,0] = 1 - 2*(q[2]**2 + q[3]**2)
    R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0,2] = 2*(q[1]*q[3] + q[0]*q[2])
    R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[1,1] = 1 - 2*(q[1]**2 + q[3]**2)
    R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2,1] = 2*(q[2]*q[3] + q[0]*q[1])
    R[2,2] = 1 - 2*(q[1]**2 + q[2]**2)
    return R