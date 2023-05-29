#Final Project - Initial Inspection
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_hdf5_items(name, obj):
    if isinstance(obj, h5py.Dataset):
        print("Dataset:", name)
        print("    Shape:", obj.shape)
        print("    Data type:", obj.dtype)
    elif isinstance(obj, h5py.Group):
        print("Group:", name)
    else:
        print("Unknown item:", name)

# Open the h5py file
file = h5py.File('data.hdf5', 'r')

file.visititems(print_hdf5_items)

# Remember to close the file when you're done
dataset = file['synced/grav']

# Sample some values from the dataset
mean = dataset[0:]


# Print the samples
print(mean)

file.close()
