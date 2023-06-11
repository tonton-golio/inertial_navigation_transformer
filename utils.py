from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import os
import pandas as pd
# to import OneHotEncoder: 
from sklearn.preprocessing import OneHotEncoder

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

def create_sequences(X, y, seq_length, overlap=1):
    """
    Function to create sequences from the input data.

    Args:
    X : numpy array
        Input data.
    y : numpy array
        Target data.
    seq_length : int
        Sequence length.
    overlap : int, optional
        Overlap between sequences. Defaults to 1.

    Returns:
    X_seq, y_seq : numpy arrays
        Sequenced input and target data.
    """
    X_seq = []
    y_seq = []
    for i in range(0, len(X) - seq_length + 1, seq_length-overlap):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def load_much_data(N_train, N_test, folder_path, columns_X, columns_y, seq_length=1, verbose=False, num_datasets=1, random_start=False, include_clusters=False, cluster_labels_path=''):
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
    
    Nloaded_points = 0
    # get control of directories
    dirs = os.listdir(folder_path)
    if '.DS_Store' in dirs: dirs.remove('.DS_Store') # remove .DS_Store if present
    dirs = sorted(dirs)
    #print(dirs)
    test_dir = dirs[:1]
    train_dirs = dirs[1:num_datasets+1]
    print(f'using {test_dir} for testing and the remaining ({len(train_dirs)}) for training')
    

    N_points_per_dir = max(int(N_train/len(train_dirs)), seq_length)
    N_points_per_dir = N_points_per_dir - N_points_per_dir % seq_length
    n_dirs_to_use = max([ 1, int(N_train/N_points_per_dir)])
    print(f'using {n_dirs_to_use} directories for training')
    train_dirs = train_dirs[:n_dirs_to_use+1]
    print('test dirs:', test_dir, 'train dirs:', train_dirs)
    N_points = N_points_per_dir * len(train_dirs)
    
    print(f'Loading a total of {N_train}, with {N_points_per_dir} points from each of {len(train_dirs)} directories')

    if include_clusters: 
        columns_X.append('cluster_labels')
        cluster_labels = {}
        for dir in dirs:
            if dir in test_dir or dir in train_dirs:
                filepath = f'{cluster_labels_path}/{dir}.csv'
                df = pd.read_csv(filepath, header=0, index_col=0)
                labels = df['birch_labels'].values.reshape(-1,1)

                # one hot encode labels
                onehot_encoder = OneHotEncoder(sparse=False)
                onehot_encoded = onehot_encoder.fit_transform(labels)
                cluster_labels[dir] = onehot_encoded

    data = {
        'X-train': {key: None for key in columns_X},
        'y-train': {key: None for key in columns_y},
        'X-test': {key: None for key in columns_X},
        'y-test': {key: None for key in columns_y},
    }
    for dir in dirs:
        print(dir)
        file_path = os.path.join(folder_path, dir, 'data.hdf5')
        if verbose: print('Loading file:', file_path)
        

        with h5py.File(file_path, "r") as hdf_file:
            new_data_dict = load_data(file_path)
            if include_clusters:
                new_data_dict['cluster_labels'] = cluster_labels[dir]
        if dir in test_dir:
            
            start = 0 if not random_start else np.random.randint(0, 20000)
            if start + N_test > len(new_data_dict[list(new_data_dict.keys())[0]]):
                start = 0
            for key in columns_X:
                if data['X-test'][key] is None:
                    data['X-test'][key] = new_data_dict[key][start:start+N_test]
                else:
                    data['X-test'][key] = np.vstack([data['X-test'][key], new_data_dict[key][:N_test]])
            for key in columns_y:
                if data['y-test'][key] is None:
                    data['y-test'][key] = new_data_dict[key][start:start+N_test]
                else:
                    data['y-test'][key] = np.vstack([data['y-test'][key], new_data_dict[key][:N_test]])
        elif dir in train_dirs:
            Nloaded_points += N_points_per_dir
            print('dir:', dir, 'is in train_dirs')
            for key in columns_X:
                if data['X-train'][key] is None:
                    data['X-train'][key] = new_data_dict[key][start:start+N_points_per_dir]
                else:
                    data['X-train'][key] = np.vstack([data['X-train'][key], new_data_dict[key][:N_points_per_dir]])
            for key in columns_y:
                if data['y-train'][key] is None:
                    data['y-train'][key] = new_data_dict[key][start:start+N_points_per_dir]
                else:
                    data['y-train'][key] = np.vstack([data['y-train'][key], new_data_dict[key][:N_points_per_dir]])
        if Nloaded_points >= N_points:
            break

    return data

def normalize_features(X, scaler=None):
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
    if scaler is None:
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
            - N_points: number of training instances
            - seq_len: sequence length, must be >= 1
            - input: list of input features, e.g. ['pose/tango_ori', 'pose/tango_pos', 'synced/gyro']
            - output: list of output features, e.g. ['pose/tango_ori']
            - normalize: boolean, whether to normalize the data or not.
            - shuffle: boolean, whether to shuffle the data or not.
            - verbose: boolean, whether to print information about the data or not.
            - include_theta: boolean, whether to include the the quat update from analytical
    Returns:
    X_train : numpy array
    y_train : numpy array
    X_test : numpy array
    y_test : numpy array



    """
    params = {'N_train': 1000,'N_test': 100, 'seq_len': 10, 'input': [], 'output': [], 'normalize': False, 'shuffle': True, 'verbose': True, 'num_datasets':1, 'random_start':True, 'include_theta':False, 'include_clusters':False, 'overlap':0, 'cluster_labels_path':'/Users/antongolles/Documents/uni/masters/myMasters/applied_machine_learning/inertial_navigation_transformer/Clustering_labels'}
    params.update(kwargs)

    allowed_columns = [
        'pose/ekf_ori', 'pose/tango_ori', 'pose/tango_pos',
        'synced/acce', 'synced/game_rv', 'synced/grav', 'synced/gyro', 'synced/gyro_uncalib', 'synced/linacce', 'synced/magnet', 'synced/rv',
        'raw/imu/acce', 'raw/imu/game_rv', 'raw/imu/gps', 'raw/imu/gravity', 'raw/imu/gyro', 'raw/imu/gyro_uncalib', 'raw/imu/linacce', 'raw/imu/magnet', 'raw/imu/magnetic_rv', 'raw/imu/pressure', 'raw/imu/rv', 'raw/imu/step', 'raw/imu/wifi_address', 'raw/imu/wifi_values',
        'raw/tango/acce', 'raw/tango/game_rv', 'raw/tango/gps', 'raw/tango/gravity', 'raw/tango/gyro', 'raw/tango/gyro_uncalib', 'raw/tango/linacce', 'raw/tango/magnet', 'raw/tango/magnetic_rv', 'raw/tango/pressure', 'raw/tango/rv', 'raw/tango/step', 'raw/tango/tango_adf_pose', 'raw/tango/tango_pose', 'raw/tango/wifi_address', 'raw/tango/wifi_values',
    ]

    


    # make sure columns are in allowed_columns
    columns = params['output'] + params['input']
    for column in columns:
        if column not in allowed_columns:
            raise NameError(f'ERROR: Column "{column}" not in allowed columns: {allowed_columns}')

    data = load_much_data(folder_path=folder_path, 
                            columns_X=params['input'], columns_y=params['output'], 
                            N_train=params['N_train'], N_test=params['N_test'],
                            verbose=params['verbose'], 
                            seq_length=params['seq_len'], 
                            num_datasets=params['num_datasets'],
                            random_start=params['random_start'],
                            include_clusters=params['include_clusters'],
                            cluster_labels_path=params['cluster_labels_path'],)
    if params['include_theta']:
        def Omega(w):
            """
            Takes w = [wx, wy, wz] which is the angular velocity readings from the gyroscope
            Returns the 4x4 matrix Omega(w) which is used to update the quaternion
            """
            return np.array([[0    , w[2], -w[1], w[0]],
                            [-w[2], 0    , w[0] , w[1]],
                            [w[1] , -w[0], 0    , w[2]],
                            [-w[0], -w[1], -w[2], 0   ]], dtype=np.float32) * .5

        def u_b(w, dt):
            """
            Approximates angle change based on gyroscope reading and timestep size.
            
            Parameters:
            w : np.array
                Gyro readings.
            dt : float
                Time step.

            Returns:
            u_b : np.array
                The angle change as obtained by the product of the angular velocity and the timestep.
            """

            return w * dt

        def Theta(w, dt):
            """
            Computes the quaternion update matrix based on gyroscope reading and timestep size.
            
            Parameters:
            ws : np.array
                Gyroscope readings.
            dt : float
                Time step.
            method : str, optional
                Integration method. Choose from 'euler', 'trapezoidal', or 'simpson'. 
                Defaults to 'euler'.

            Returns:
            Theta : np.array
                The quaternion update matrix (4x4).
            """
            u = u_b(w, dt)
            w_norm = vec_norm(w)
            W = Omega(w)
            u_b_norm = vec_norm(u)

            return np.cos(u_b_norm / 2) * np.eye(4) + (np.sin(u_b_norm / 2) / w_norm) * W

        dt = 1/200
        thetas_test = []
        thetas_train = []
        for i in range(data['X-train']['synced/gyro'].shape[0]):
            thetas_train.append(Theta(data['X-train']['synced/gyro'][i], dt).flatten())

        for i in range(data['X-test']['synced/gyro'].shape[0]):
            thetas_test.append(Theta(data['X-test']['synced/gyro'][i], dt).flatten())

        thetas_train = np.array(thetas_train)
        thetas_test = np.array(thetas_test)

        data['X-train']['theta'] = thetas_train
        data['X-test']['theta'] = thetas_test

        params['input'].append('theta')



    col_locations = {
        'input' : {},
        'output' : {}
    }
    counta_input = 0
    counta_output = 0
    for _, col in enumerate(params['input']):
        width = data['X-train'][col].shape[1]
        col_locations['input'][col] = (counta_input, counta_input+width)
        counta_input += width
    for _, col in enumerate(params['output']):
        width = data['y-train'][col].shape[1]
        col_locations['output'][col] = (counta_output, counta_output+width)
        counta_output += width
    

    X_train = np.hstack([data['X-train'][key] for key in params['input']])
    y_train = np.hstack([data['y-train'][key] for key in params['output']])
    X_test = np.hstack([data['X-test'][key] for key in params['input']])
    y_test = np.hstack([data['y-test'][key] for key in params['output']])
    if params['normalize']:
        X_train, scaler = normalize_features(X_train)
        # use the same scaler for test data
        X_test, _ = normalize_features(X_test, scaler=scaler)

        
    X_train_reshaped, y_train_reshaped = create_sequences(X_train, y_train, 
                                                          seq_length=params['seq_len'],
                                                          overlap=params['overlap'],
                                                          )
    X_test_reshaped, y_test_reshaped = create_sequences(X_test, y_test, 
                                                        seq_length=params['seq_len'], 
                                                        overlap=params['overlap'])

    return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped, col_locations

def split_data(X, y, test_size=0.2):
    indicies = np.arange(X.shape[0])
    indicies_train = indicies[:int((1-test_size)*X.shape[0])]
    indicies_test = indicies[int((1-test_size)*X.shape[0]):]
    X_train, X_test, y_train, y_test = X[indicies_train], X[indicies_test], y[indicies_train], y[indicies_test]
    return X_train, X_test, y_train, y_test

def calculate_position_difference(y):
    for i in range(len(y)):
        y[i] -= y[i][0]
    return y
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