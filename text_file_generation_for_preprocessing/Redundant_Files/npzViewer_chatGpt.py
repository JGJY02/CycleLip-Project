import numpy as np

# load the .npz file
data = np.load('datasets/may/audio')

# get the names of the arrays
array_names = data.files

# extract the arrays from the .npz file
for name in array_names:
    array = data[name]
    print(name, array.shape)

# close the .npz file
data.close()