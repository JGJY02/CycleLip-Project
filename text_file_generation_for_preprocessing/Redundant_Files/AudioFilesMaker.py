import os
import pathlib
import argparse



parser = argparse.ArgumentParser(description='Find the audio files and provide the directories to them based on the original')
parser.add_argument('--main_dir', type=str, required=True, help='path to the directory containing the images')
parser.add_argument('--output_dir', type=str, required=True, help='path to the directory containing the images')
args = parser.parse_args()

with open(os.path.join(args.output_dir, 'train_audio.txt'), 'w') as f:
    for dirpath, dirnames, filenames in os.walk(args.main_dir):
        for filename in filenames:
            # Check if the file extension is in the audio_extensions list
            if filename.endswith('.wav'):
                
                basename, _ = os.path.splitext(filename) #First extract just file name no extension
                relativePath = os.path.relpath(dirpath, args.main_dir)

                f.write(os.path.join(relativePath, basename) + '\n')


with open(os.path.join(args.output_dir, 'train_face.txt'), 'w') as f:
    for dirpath, dirnames, filenames in os.walk(args.main_dir):
        for filename in filenames:
            # Check if the file extension is in the audio_extensions list
            if filename.endswith('.h5'):
                relativePath = os.path.relpath(dirpath, args.main_dir)

                f.write(os.path.join(relativePath) + '\n')