# load data
import sys 
if '..' not in sys.path:
    sys.path.append('../')
from utils import *

# data loading
folder_path = '/Users/antongolles/Documents/uni/masters/myMasters/applied_machine_learning/inertial_navigation_transformer/data/data_from_RoNIN/train_dataset_1/'
params = {'Ntrain': 80_000, 'Nval': 100, 'seq_len': 8, 
          'input': ['synced/acce', 'synced/magnet', 'synced/gyro'], 
          'output': ['pose/ekf_ori'], 
          'normalize': False,
          'include_clusters':True,
          }
X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped, col_locations = load_split_data(folder_path, **params)
print(col_locations)
print('X_train_reshaped.shape: ', X_train_reshaped.shape)
print('y_train_reshaped.shape: ', y_train_reshaped.shape)
print('X_test_reshaped.shape: ', X_test_reshaped.shape)
print('y_test_reshaped.shape: ', y_test_reshaped.shape)

print(X_train_reshaped[0,0,:])

print()
print()
print()
print()
print()