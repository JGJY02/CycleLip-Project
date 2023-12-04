import os
import random
import argparse
import numpy as np
import shutil
from tqdm import tqdm



# Define the command line arguments
parser = argparse.ArgumentParser(description='Deletion of excess Preprocessed Data')
parser.add_argument('--preprocessed_dir', type=str, required=True, help='path to remove lists')
parser.add_argument('--files_to_keep', type = int, required = True, help = 'Total number of files to keep' )

args = parser.parse_args()

# Get the list of all image filenames in the directory
all_audio = []

input_dir, name = os.path.split(args.preprocessed_dir)

with open(os.path.join(input_dir, name+'_info' ,name+'.txt'), 'r') as file:
    paths_to_delete = file.readlines()
    paths_to_delete = [path.strip() for path in paths_to_delete]
    paths_to_delete = paths_to_delete[args.files_to_keep : ]

print("Operations Starting")
for path in tqdm(paths_to_delete, desc="Deleting Files"):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    else:
        continue

print("Deletion Complete! \n")
print("Total amount of files removed is ", len(paths_to_delete))        

