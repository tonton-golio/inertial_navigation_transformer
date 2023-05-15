import h5py
import matplotlib.pyplot as plt
import numpy as np


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



# we need a cool dataset class, that can be used for training
# we should take a batch of data like 1000 timesteps, but perhaps they should not all be this length
# we can fill zeros at the end. 




# Kalman filter for gyro: https://sci-hub.hkvisa.net/10.3390/s111009182
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