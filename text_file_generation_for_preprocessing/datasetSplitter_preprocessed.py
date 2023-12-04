from fileinput import filename
import os
import random
import argparse
import numpy as np
# Define the command line arguments
parser = argparse.ArgumentParser(description='Split images into train, test, and validation sets.')
parser.add_argument('--video_dir', type=str, required=True, help='path to the directory containing the txt')
parser.add_argument('--output_dir', type=str, required=True, help='path to the directory to save the text files')
parser.add_argument('--specify_vid_count', type=int, default = 0, help = 'Use all or specify vid, 1 for specify 0 for all' )
parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of images to use for the train set (default: 0.8)')
parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of images to use for the validation set (default: 0.1)')
parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of images to use for the test set (default: 0.1)')
parser.add_argument('--num_of_videos', type=int, default = 900)
args = parser.parse_args()

test = True
# Get the list of all image filenames in the directory
all_videos_directories = []
for dirpath, dirnames, filenames in os.walk(args.video_dir):
    if os.path.relpath(dirpath, args.video_dir) != '.':
        if test:
            print(os.path.relpath(dirpath, args.video_dir))
            test = False

        for dirname in dirnames:
            all_videos_directories.append(os.path.join(dirpath,dirname))

with open(os.path.join(args.output_dir, 'all.txt'), 'w') as f:
    for vid_all in all_videos_directories:
        f.write(vid_all + '\n')

# # Shuffle the list of image filenames randomly
random.shuffle(all_videos_directories)

# Calculate the number of images to use for each set
if(args.specify_vid_count == 1):
     num_audio = args.num_of_videos
else:
    num_audio = len(all_videos_directories)

num_train = int(num_audio * args.train_ratio)
num_val = int(num_audio * args.val_ratio)
num_test = num_audio - num_train - num_val

# Create separate lists of image filenames for each set
train_files = all_videos_directories[:num_train]
val_files = all_videos_directories[num_train:num_train+num_val]
test_files = all_videos_directories[num_train+num_val:]

# Write the filenames for each set to separate text files
with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
    for vid_train in train_files:
        f.write(vid_train + '\n')

with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
    for vid_val in val_files:
        f.write(vid_val + '\n')

with open(os.path.join(args.output_dir, 'test.txt'), 'w') as f:
    for vid_test in test_files:
        f.write(vid_test + '\n')

