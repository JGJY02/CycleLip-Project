import os
import random
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob

# Define the command line arguments
parser = argparse.ArgumentParser(description='Obtain all videos within dataset')
parser.add_argument('--preprocessed_dir', type=str, required=True, help='path to the directory containing preprocessed files')

args = parser.parse_args()


output_dir, name = os.path.split(args.preprocessed_dir)
# Get the list of all image filenames in the directory
preprocessed_folders = []
print("Obtaining preprocessed files")

for directory in tqdm(os.listdir(args.preprocessed_dir), desc='Main Directories Parsed'):
    directory_path = os.path.join(args.preprocessed_dir, directory)
        
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)

        for viddir in os.listdir(subdir_path):
            vid_path = os.path.join(directory, subdir, viddir)
            preprocessed_folders.append(vid_path)


list = os.path.join(output_dir, name+'_info',name + '_wav2lip.txt')
list_info = os.path.join(output_dir, name+'_info',name + '_wav2lip_info.txt')



## Generate the list of preprocessed videos
with open(list, 'w') as f:
    for directory in preprocessed_folders:
        f.write(directory + '\n')

with open(list_info, 'w') as f:
    f.write("Number of Videos: " + str(len(preprocessed_folders)) + '\n')

print("All video files list generated")