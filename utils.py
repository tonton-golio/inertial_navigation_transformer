from collections import defaultdict
from functools import partial
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import os

def get_all_datasets(hdf_file):
    """
    Function to return all datasets from an HDF5 file.

    Args:
    hdf_file : h5py.File object

    Returns:
    datasets : dict
        Dictionary with dataset names as keys and numpy arrays as values.
    """
    datasets = {}

    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj[:]

    hdf_file.visititems(collect_datasets)
    return datasets

def nice_dict_contents(data_dict, print_keys=False):
    """
    Function to print the contents of the dictionary in a hierarchical manner.
    
    Args:
    data_dict : dict
        Dictionary containing the data.
    print_keys : bool, optional
        Flag indicating whether to print the keys. Defaults to False.
    """
    outer_keys = defaultdict(lambda: defaultdict(list))
    
    for key in data_dict.keys():
        split_key = key.split('/')
        
        if len(split_key) == 2:
            outer, inner = split_key
            outer_keys[outer][inner]
        else:
            outer, middle, inner = split_key
            outer_keys[outer][middle].append(inner)
            
    if print_keys:
        print('CONTENTS OF HDF5 FILE:')
        for outer_key, outer_value in outer_keys.items():
            print(outer_key)
            for inner_key, inner_values in outer_value.items():
                print('\t', inner_key)
                print('\t\t', ', '.join(inner_values))

def load_data(file_path, verbose=False):
    """
    Function to load the data from a given file path.

    Args:
    file_path : str
        Path to the data file.
    verbose : bool, optional
        If True, print the contents of the file.

    Returns:
    data_dict : dict
        Dictionary with dataset names as keys and numpy arrays as values.
    """
    with h5py.File(file_path, 'r') as hdf_file:
        data_dict = get_all_datasets(hdf_file)
    
    if verbose:
        nice_dict_contents(data_dict, print_keys=True)

    return data_dict

def create_sequences(X, y, seq_length):
    """
    Function to create sequences from the input data.

    Args:
    X : numpy array
        Input data.
    y : numpy array
        Target data.
    seq_length : int
        Sequence length.

    Returns:
    X_seq, y_seq : numpy arrays
        Sequenced input and target data.
    """
    X_seq = [X[i:i+seq_length] for i in range(0, len(X) - seq_length + 1, seq_length)]
    y_seq = [y[i:i+seq_length] for i in range(0, len(y) - seq_length + 1, seq_length)]
    return np.array(X_seq), np.array(y_seq)

def load_much_data(Ntrain, Nval, folder_path, columns):
    """
    Function to load data from multiple HDF5 files.

    Args:
    Ntrain : int
        Number of training instances.
    Nval : int
        Number of validation instances.
    folder_path : str
        Path to the data directory.
    columns : list
        List of column names.

    Returns:
    data_dict : dict
        Dictionary with dataset names as keys and numpy arrays as values.
    """
    data_dict = {key: None for key in columns}
    Nloaded_points = 0
    
    for root, dirs, files in os.walk(folder_path):
        if Nloaded_points > Ntrain + Nval:
            break
        for file in files:
            if file.endswith(".hdf5"):
                file_path = os.path.join(root, file)
                with h5py.File(file_path, "r") as hdf_file:
                    new_data_dict = load_data(file_path)
                    Nloaded_points += new_data_dict[columns[0]].shape[0]
                # add values to data_dict
                for key in columns:
                    if data_dict[key] is None:
                        data_dict[key] = new_data_dict[key]
                    else:
                        data_dict[key] = np.vstack([data_dict[key], new_data_dict[key]])
    # Ensure we have Ntrain+Nval entries
    for key in columns:
        data_dict[key] = data_dict[key][:Ntrain + Nval]
    return data_dict

def normalize_features(X):
    """
    Function to normalize the features.

    Args:
    X : numpy array
        Input data.

    Returns:
    X_normalized : numpy array
        Normalized input data.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used for normalization. Useful for inverse transformation.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler

def load_split_data(folder_path='C:\\Users\\Simon Andersen\\Documents\\Uni\\KS6\\AppliedML\\Project 2\\train_dataset_1', **kwargs):
    """
    Function to load, split, and preprocess the data.

    Args:
    folder_path : str
        Path to the data directory.
    **kwargs : other parameters to control the data loading and processing.
            - Ntrain: number of training instances
            - Nval: number of validation instances
            - seq_len: sequence length, must be > 1
            - input: list of input features, e.g. ['pose/tango_ori', 'pose/tango_pos', 'synced/gyro']
            - output: list of output features, e.g. ['pose/tango_ori']
            - normalize: boolean, whether to normalize the data or not.

    Returns:
    X_reshaped : numpy array
        The processed and reshaped input data.
    y_reshaped : numpy array
        The processed and reshaped target data.
    """
    params = {'Ntrain': 1000, 'Nval': 100, 'seq_len': 10, 'input': [], 'output': [], 'normalize': False}
    params.update(kwargs)

    allowed_columns = [
        'pose/ekf_ori', 'pose/tango_ori', 'pose/tango_pos',
        'synced/acce', 'synced/game_rv', 'synced/grav', 'synced/gyro', 'synced/gyro_uncalib', 'synced/linacce', 'synced/magnet', 'synced/rv',
        'raw/imu/acce', 'raw/imu/game_rv', 'raw/imu/gps', 'raw/imu/gravity', 'raw/imu/gyro', 'raw/imu/gyro_uncalib', 'raw/imu/linacce', 'raw/imu/magnet', 'raw/imu/magnetic_rv', 'raw/imu/pressure', 'raw/imu/rv', 'raw/imu/step', 'raw/imu/wifi_address', 'raw/imu/wifi_values',
        'raw/tango/acce', 'raw/tango/game_rv', 'raw/tango/gps', 'raw/tango/gravity', 'raw/tango/gyro', 'raw/tango/gyro_uncalib', 'raw/tango/linacce', 'raw/tango/magnet', 'raw/tango/magnetic_rv', 'raw/tango/pressure', 'raw/tango/rv', 'raw/tango/step', 'raw/tango/tango_adf_pose', 'raw/tango/tango_pose', 'raw/tango/wifi_address', 'raw/tango/wifi_values',
    ]

    
    try:
        Ntrain, Nval = params['Ntrain'], params['Nval']
        seq_len = params['seq_len']
        input_features = params['input']
        output_features = params['output']
        columns = input_features + output_features
        
        # make sure columns are in allowed_columns
        for column in columns:
            if column not in allowed_columns:
                raise NameError(f'ERROR: Column "{column}" not in allowed columns: {allowed_columns}')

        data_dict = load_much_data(Ntrain, Nval, folder_path=folder_path, columns=columns)

        X = np.hstack([data_dict[key] for key in input_features])
        
        if params['normalize']:
            X, _ = normalize_features(X)
            
        y = np.hstack([data_dict[key] for key in output_features])
        X_reshaped, y_reshaped = create_sequences(X, y, seq_length=seq_len)

        return X_reshaped, y_reshaped
    except KeyError as e:
        print(f'Missing necessary parameter: {e}')
    except NameError as e:
        print(e)

    

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