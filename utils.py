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

