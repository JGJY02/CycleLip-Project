import cv2
import face_alignment
import numpy as np
import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import subprocess
import shutil
import time



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inp_dir", required=True, help='input directory with .mp4 videos')
    parser.add_argument("--out_dir", type=str, default='.', help='file path with currently processed files')
    parser.add_argument("--raw_file", type=str, default='.', help='filepath to output the remaining data')
    parser.add_argument("--remaining_file", type=str, default='.', help='filepath to output the remaining data')
    args = parser.parse_args()

    All_videopaths = []

    if not os.path.exists(args.raw_file):
        #Obtain the sub directories for the input
        print("Obtaining unprocessed files")
        unfiltered_subdirs = [folder[0] for folder in os.walk(args.inp_dir)]
        unfiltered_subdirs.pop(0) # remove the first as that is the main directory

        subdirs = [subfolder for subfolder in unfiltered_subdirs if len(list(glob(os.path.join(subfolder, '*.mp4')))) != 0]
        print("Total number of subdirectories %d" % (len(subdirs)) )

        for subfolder in subdirs:
            All_videopaths.append(os.path.relpath(subfolder,args.inp_dir))

        with open(args.raw_file, 'w') as f:
            for directory in subdirs:
                f.write(directory + '\n')


    
    else:
        with open(args.raw_file, 'r') as f:
            # read the contents of the file into a string variable
            All_videopaths = [os.path.relpath(path.rstrip('\n'), args.inp_dir) for path in f.readlines()] #to read lines and remove the \n which gets added via f.readlines

    #Obtain the sub directories for the output
    print("Obtaining processed folders")
    out_unfiltered_subdirs = [folder[0] for folder in os.walk(args.out_dir)]
    out_unfiltered_subdirs.pop(0) # remove the first as that is the main directory

    out_subdirs = [os.path.dirname(os.path.relpath(out_subfolder,args.out_dir)) for out_subfolder in out_unfiltered_subdirs if len(list(glob(os.path.join(out_subfolder, '*.npz')))) and len(list(glob(os.path.join(out_subfolder, '*.npy')))) and len(list(glob(os.path.join(out_subfolder, '*.wav')))) != 0]


    #For optimization (Suggested by chat GPT)
    out_subdirs_set = set(out_subdirs)
    updatedlist = [path for path in All_videopaths if path not in out_subdirs_set]

    
    #Remove duplicate subfolders
    updatedlist = list(set(updatedlist))
    print("Total number of subdirectories %d" % (len(updatedlist)) )

    num_videos = len(updatedlist)
    num_split = int(np.ceil(num_videos/5))
    
    #Write the remaining files into a txt file
    with open(os.path.join(args.remaining_file, 'RemainingFiles.txt'), 'w') as f:
        for updatedPaths in updatedlist:
            f.write(os.path.join(args.inp_dir, updatedPaths) + '\n')

    with open(os.path.join(args.remaining_file, 'RemainingFiles_A.txt'), 'w') as f:
        for updatedPaths in updatedlist[0:num_split]:
            f.write(os.path.join(args.inp_dir, updatedPaths) + '\n')
    
    with open(os.path.join(args.remaining_file, 'RemainingFiles_B.txt'), 'w') as f:
        for updatedPaths in updatedlist[num_split:num_split*2]:
            f.write(os.path.join(args.inp_dir, updatedPaths) + '\n')

    with open(os.path.join(args.remaining_file, 'RemainingFiles_C.txt'), 'w') as f:
        for updatedPaths in updatedlist[num_split*2:num_split*3]:
            f.write(os.path.join(args.inp_dir, updatedPaths) + '\n')
    
    with open(os.path.join(args.remaining_file, 'RemainingFiles_D.txt'), 'w') as f:       
        for updatedPaths in updatedlist[num_split*3:num_split*4]:
            f.write(os.path.join(args.inp_dir, updatedPaths) + '\n')
    
    with open(os.path.join(args.remaining_file, 'RemainingFiles_E.txt'), 'w') as f:
        for updatedPaths in updatedlist[num_split*4:]:
            f.write(os.path.join(args.inp_dir, updatedPaths) + '\n')