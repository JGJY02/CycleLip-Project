import os
import random
import argparse
import h5py
import numpy as np
# Define the command line arguments
parser = argparse.ArgumentParser(description='Split images into train, test, and validation sets.')
parser.add_argument('--video_dir', type=str, required=True, help='path to the directory containing the images')
parser.add_argument('--output_dir', type=str, required=True, help='path to the directory to save the text files')
parser.add_argument('--specify_vid_count', type=int, default = 0, help = 'Use all or specify vid, 1 for specify 0 for all' )
parser.add_argument('--num_of_videos', type=int, default = 5000, help = 'define the number of video required for training' )
parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of images to use for the train set (default: 0.8)')
parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of images to use for the validation set (default: 0.1)')
parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of images to use for the test set (default: 0.1)')
parser.add_argument('--splits', type=int, default=10, help='How many txt files to preprocess all videos')

args = parser.parse_args()

# Get the list of all image filenames in the directory
all_videos = []

with open(args.video_dir, 'r') as f:
    # read the contents of the file into a string variable
    all_videos = [path.rstrip('\n') for path in f.readlines()] #to read lines and remove the \n which gets added via f.readlines

# # Shuffle the list of image filenames randomly
random.shuffle(all_videos)

# Calculate the number of images to use for each set
if(args.specify_vid_count == 1):
     num_audio = args.num_of_videos
else:
    num_audio = len(all_videos)
num_train = int(num_audio * args.train_ratio)
num_val = int(num_audio * args.val_ratio)
num_test = num_audio - num_train - num_val

# Create separate lists of image filenames for each set
train_audio = all_videos[:num_train]
val_audio = all_videos[num_train:num_train+num_val]
test_audio = all_videos[num_train+num_val:]

# Write the filenames for each set to separate text files





with open(os.path.join(args.output_dir, 'allsplit.txt'), 'w') as f:
    for audio in all_videos[0:num_audio]:
        f.write(os.path.join(audio) + '\n')

#Train txt along with splitt files
num_of_train_vids = int(num_train / args.splits)

with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
    for audio in train_audio:
        f.write(os.path.join(audio) + '\n')

for i in range(args.splits):
    preprocessFile = os.path.join(args.output_dir, 'allsplit_train_' + str(i) + '.txt')
    with open(preprocessFile, 'w') as f:
        if(i != args.splits-1):
            for audio in all_videos[num_of_train_vids*i:num_of_train_vids*i + num_of_train_vids]:
                f.write(os.path.join(audio) + '\n')
        else:
            for audio in all_videos[num_of_train_vids*i:]:
                f.write(os.path.join(audio) + '\n')

#Val txt along with split files
num_of_val_vids = int(num_val / args.splits)

with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
    for audio in val_audio:
        f.write(os.path.join(audio) + '\n')

for i in range(args.splits):
    preprocessFile = os.path.join(args.output_dir, 'allsplit_val_' + str(i) + '.txt')
    with open(preprocessFile, 'w') as f:
        if(i != args.splits-1):
            for audio in all_videos[num_of_val_vids*i:num_of_val_vids*i + num_of_val_vids]:
                f.write(os.path.join(audio) + '\n')
        else:
            for audio in all_videos[num_of_val_vids*i:]:
                f.write(os.path.join(audio) + '\n')

#Test txt along with split files
num_of_test_vids = int(num_test / args.splits)

with open(os.path.join(args.output_dir, 'test.txt'), 'w') as f:
    for audio in val_audio:
        f.write(os.path.join(audio) + '\n')

for i in range(args.splits):
    preprocessFile = os.path.join(args.output_dir, 'allsplit_test_' + str(i) + '.txt')
    with open(preprocessFile, 'w') as f:
        if(i != args.splits-1):
            for audio in all_videos[num_of_test_vids*i:num_of_test_vids*i + num_of_test_vids]:
                f.write(os.path.join(audio) + '\n')
        else:
            for audio in all_videos[num_of_test_vids*i:]:
                f.write(os.path.join(audio) + '\n')