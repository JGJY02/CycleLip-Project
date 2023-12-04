import cv2
import face_alignment
import numpy as np
import os
from argparse import ArgumentParser
from glob import glob
import subprocess
import shutil
import h5py

def create_directories(inp, out_dir, subfolder):
    # Create a directory to save the images
    _, basename = os.path.split(inp)
    section, subfolder2 = os.path.split(subfolder)
    _, subfolder1 = os.path.split(section)
    
    # #Aligned frames directory
    # alignedframes_dir = os.path.join(
    #     out_dir,subfolder,
    #     os.path.splitext(basename)[0] + '_alignedframes'
    # )
    # os.makedirs(alignedframes_dir, exist_ok=True)

    #Regular Frames Directory
    frames_dir = os.path.join(
        out_dir, subfolder1, subfolder2, os.path.splitext(basename)[0],
        os.path.splitext(basename)[0] + '_frames'
    )

    #output directory
    output_dir = os.path.join(
        out_dir, subfolder1, subfolder2, os.path.splitext(basename)[0],
        os.path.splitext(basename)[0] + '_datafiles'
    )

    #candidate directory
    candidate_dir = os.path.join(
        output_dir, 'candidates'
    )

    return frames_dir, output_dir, candidate_dir



def extract_frames_audio(inp, fps, frames_dir, output_dir, sampleRate):
    success = True

    try:
        cmd = f"ffmpeg -y -i {inp} -vf fps={fps} {frames_dir}/%04d.jpg -hide_banner -v 0"
        process = subprocess.run(cmd, shell=True, check=True)
        if process.returncode is None:
            raise Exception("ffmpeg failed with error code {}".format(process.returncode))
    

    except subprocess.TimeoutExpired:
        # Handle the case where the command times out
        print("ffmpeg timed out")
        success = False
        return success

    except subprocess.subprocess.CalledProcessError:
        print("An error has occured while generating the image clips")
        success = False
        return success
    
    try:
        cmd = f"ffmpeg -y -i {inp} -vn -acodec pcm_s16le -ar {sampleRate} -ac 2 {output_dir}/audio.wav"
        process = subprocess.run(cmd, shell=True, check=True)
        if process.returncode is None:
            raise Exception("ffmpeg failed with error code {}".format(process.returncode))
    
    except subprocess.TimeoutExpired:
        # Handle the case where the command times out
        print("ffmpeg timed out")
        success = False
        return success
    
    except subprocess.subprocess.CalledProcessError:
        print("An error has occured while producing the audio")
        success = False
        return success

    return success


    

# def export_files(inp, fps, out_dir):
#     _, basename = os.path.split(inp)
#     frames_dir = os.path.join(
#         out_dir,
#         os.path.splitext(basename)[0] + '_frames'
#     )
#     os.makedirs(frames_dir, exist_ok=True)

#     Fit_data

#     mean_pts3d

#     tracked3d_normalized_pts

#     return Fit_data

def rotation_translation_obtainer(landmarks): #taken from  https://gist.github.com/zalo/71fbd5dbfe23cb46406d211b84be9f7e

    #Prepare Orientation matrix
    orientation = np.zeros((1,3))
    translation = np.zeros((1,3))

    #obtain file name
    # filename = Path(framepath).stem
    # out_dir = os.path.join(alignedframes_dir, filename + '.jpg')
    
    # Compute the Mean-Centered-Scaled Points 
    # (Essentially what we are doing here is that we are first trying to find the average position of the landmarks and based on this we will try to find the center of the face. 
    # With this we find our positions relative to it)

    # mean = np.mean(landmarks, axis=0) # <- This is the unscaled mean
    scaled = (landmarks / np.linalg.norm(landmarks[42] - landmarks[39])) * 0.06 # Set the inner eye distance to 60mm (This is due to the average distance being 60-62mm) https://en.wikipedia.org/wiki/Telecanthus#:~:text=In%20most%20people%2C%20the%20intercanthal,of%20approximately%2030%E2%80%9331%20mm.
    centered = scaled - np.mean(scaled, axis=0) # <- This is the scaled mean

    # # Construct the translation matrix (Here we find the translations relative to the center)
    translation[0,0] = np.mean((landmarks[:,0] - centered[:,0])) # Obtain the average x translation
    translation[0,1] = np.mean((landmarks[:,1] - centered[:,1])) # Obtain the average y translation
    translation[0,2] = np.mean((landmarks[:,2] - centered[:,2])) # Obtain the average z translation

    # Construct a "rotation" matrix (strong simplification, might have shearing)
    rotationMatrix = np.empty((3,3))
    rotationMatrix[0,:] = (centered[16] - centered[0])/np.linalg.norm(centered[16] - centered[0]) # 
    rotationMatrix[1,:] = (centered[8] - centered[27])/np.linalg.norm(centered[8] - centered[27])
    rotationMatrix[2,:] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])
    invRot = np.linalg.inv(rotationMatrix)
    
    # Object-space points, these are what you'd run OpenCV's solvePnP() with
    # objectPoints = centered.dot(invRot)

    #Calculate the roll pitch and yaw
    orientation[0,2]=np.degrees(np.arctan2(rotationMatrix[1,0],rotationMatrix[0,0])) #z Rotation roll
    orientation[0,1]=np.degrees(np.arctan2((-1*rotationMatrix[2,0]),np.sqrt(rotationMatrix[2,1]**2+rotationMatrix[2,2]**2))) #y rotation yaw
    orientation[0,0]=np.degrees(np.arctan2(rotationMatrix[2,1],rotationMatrix[2,2])) #x rotation pitch

    # Draw the computed data (This will help for checking the frames and orientation to see how well the model has tracked the points) #Can be commented out as this is not necessary
    # for i, (imagePoint, objectPoint) in enumerate(zip(landmarks, objectPoints)):

        
    #     # Draw the Point Predictions
    #     cv2.circle(frame, (int(imagePoint[0]), int(imagePoint[1])), 3, (0,255,0))

    #     # Draw the X Axis
    #     cv2.line(frame, tuple(mean[:2].astype(int)), 
    #                     tuple((mean+(rotationMatrix[0,:] * 100.0))[:2].astype(int)), (0, 0, 255), 3)
    #     # Draw the Y Axis
    #     cv2.line(frame, tuple(mean[:2].astype(int)), 
    #                     tuple((mean-(rotationMatrix[1,:] * 100.0))[:2].astype(int)), (0, 255, 0), 3)
    #     # Draw the Z Axis
    #     cv2.line(frame, tuple(mean[:2].astype(int)), 
    #                     tuple((mean+(rotationMatrix[2,:] * 100.0))[:2].astype(int)), (255, 0, 0), 3)
        
    # cv2.imwrite(out_dir, frame)
    
    return orientation, translation
    
    
def remove_folders(frames_dir, landmark_exist = True, output_dir = None):

    shutil.rmtree(frames_dir)

    if(landmark_exist == False):
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inp_dir", required=True, help='input directory with .mp4 videos')
    parser.add_argument("--image_shape", default=512, type=int,
                        help="Image shape")
    parser.add_argument("--fps", dest="fps", type=int, help="fps", default=60)
    parser.add_argument("--out_dir", type=str, default='.')
    parser.add_argument("--device", type=str, default = 'cpu', help="cpu mode.")
    parser.add_argument("--points", type=int, help="Number of facial landmarks used", default = 68)
    parser.add_argument("--sampleRate", type=int, help="Sampling rate for audio", default = 16000)
    parser.add_argument("--txt_list", type=str, help="use the list to preprocess the remaining data")

    args = parser.parse_args()

    subFolderCount = 0
    
    
    with open(args.txt_list+'.txt', 'r') as f:
        # read the contents of the file into a string variable
        subdirs = [path.rstrip('\n') for path in f.readlines()] #to read lines and remove the \n which gets added via f.readlines
    
    print("Total number of subdirectories %d" % (len(subdirs)) )
    
    #Initialize the detector
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=args.device, flip_input=False)
    fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, flip_input=False)
    
    #Initialize mouth indices
    innerMouthIndexes = [60, 61, 62, 63, 64, 65, 66, 67]

    for subfolder in subdirs:
        VideoPaths = list(glob(os.path.join(subfolder, '*.mp4')))

        subFolderCount += 1

        print("On subfolder %d / %d" %(subFolderCount, len(subdirs)))
        print("Total video paths are %d" %(len(VideoPaths)))

        for video in VideoPaths:
            landmark_exist = True         
            curRow = 0
            newvideo = True
            image_paths = []
            Mouth_areas = []
            compiled_candidate_Images = []

            frames_dir, output_dir, candidate_dir = create_directories(video, args.out_dir, subfolder)
            
            #This will skip over processing that has already been completed
            if os.path.exists(os.path.join(output_dir+'/3d_fit_data.npz')):
                 print("File path already exists skip to next video")
                 continue
            
            else:
                print("Video Preprocessing has not been done Perform new analysis")
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(candidate_dir, exist_ok=True)
            
            # This is a log to tell us what videos have already been Processed
            with open(os.path.join(args.txt_list+'_ProcessedVideos.txt'), 'a') as f:
                f.write(os.path.relpath(video,args.inp_dir) + '\n')

            extractionSuccess = extract_frames_audio(video, args.fps ,frames_dir, output_dir, args.sampleRate)

            if(extractionSuccess == False):
                print("error occured during extraction skipping to next video")
                continue
            
            frames = list(glob(os.path.join(frames_dir, '*.jpg')))
            

            num_of_frames = len(frames)
            Compiled_landmarks = np.zeros((num_of_frames,args.points, 3))
            Compiled_orientation = np.zeros((num_of_frames, 3))
            Compiled_translation = np.zeros((num_of_frames, 3, 1))
            Compiled_landmarks_2D = np.zeros((num_of_frames,args.points, 2))

            for curFramePath in frames:
                #Append the image paths to make to list
                image_paths.append(curFramePath)

                #Read the image path and obtain the image
                curFrame = cv2.imread(curFramePath)
                
                #If this a new video write a new txt file of all the training images, if appened to it
                if(newvideo == False):
                    with open(os.path.join( output_dir, 'train.txt'), 'a') as f:
                            f.write(os.path.relpath(curFramePath,args.out_dir) + '\n')  
                else:
                    with open(os.path.join( output_dir, 'train.txt'), 'w') as f:
                            f.write(os.path.relpath(curFramePath,args.out_dir) + '\n')
                    newvideo = False     

                # #Obtain landmarks both 3d and 2d
                landmark_3d = fa_3d.get_landmarks(curFrame)
                landmark_2d = fa_2d.get_landmarks(curFrame)

                #If any frame cant be analysed drop the video and move to next
                if(np.any(landmark_3d == None) or np.any(landmark_2d == None)):
                    print("Facial landmarks are invalid video is thus invalid")
                    landmark_exist = False
                    break

                landmark_3d = np.array(landmark_3d)[0]
                landmark_2d = np.array(landmark_2d)[0]

                orientation, translation = rotation_translation_obtainer(landmark_3d)
                
                #obtain the mouth area Assume its a polygon being formed and use the shoelace method to obtain said area
                area = np.sum(landmark_2d[innerMouthIndexes][0][:-1]*landmark_2d[innerMouthIndexes][1][1:] - landmark_2d[innerMouthIndexes][1][:-1]*landmark_2d[innerMouthIndexes][0][1:])/2
                Mouth_areas.append(area)

                #Compile data
                Compiled_landmarks[curRow,:,:] = landmark_3d.reshape((1,args.points,3))
                Compiled_orientation[curRow,:] = orientation
                Compiled_translation[curRow,:,0] = translation
                Compiled_landmarks_2D[curRow,:,:] = landmark_2d.reshape((1,args.points,2))

                #Required obtain compilation of translation and fixed contours

                curRow += 1

       
            #Skip to next video if this video is invalid
            if(not landmark_exist):
                #Remove Data
                remove_folders(frames_dir, landmark_exist, output_dir)

                #This is a log to tell us which videos failed 
                with open(os.path.join(args.txt_list+'_invalidVideos.txt'), 'a') as f:
                        f.write(os.path.relpath(video,args.inp_dir) + '\n')
                continue

           # Find candidate image set
            minArea = np.array(Mouth_areas).argmin()
            maxArea = np.array(Mouth_areas).argmax()
            maxRotx = Compiled_orientation[:,0].argmax()
            maxRoty = Compiled_orientation[:,1].argmax()

            compiled_candidate_Images = [frames[minArea], frames[maxArea], frames[maxRotx], frames[maxRoty]]

            for ind in range(4):
                 candidateImages = compiled_candidate_Images[ind]
                 candidate_image_dest = os.path.join(candidate_dir, f'full_{ind}.jpg')
                 shutil.copy(candidateImages, candidate_image_dest)

           #Generate h5 File
            frames = [cv2.imread(path) for path in image_paths]

            with h5py.File(os.path.join(output_dir + '/video_frames.h5'), 'w') as hf:
                # create dataset for storing video frames
                video_data = hf.create_dataset('video_frames', data=frames, dtype=np.uint8)
        
           #Obtain the average position x,y,z of each point throughout time
            Mean_3d_Position = np.average(Compiled_landmarks, axis = 0)
            Mean_3d_Position = Mean_3d_Position.reshape(args.points, 3)

            #File names
            fit_3d_dir = os.path.join(output_dir+'/3d_fit_data.npz')
            mean_dir = os.path.join(output_dir+'/mean_pts3d.npy')
            tracked2D_normalized_pts_fix_contour = os.path.join(output_dir+'/tracked2D_normalized_pts_fix_contour.npy')

            #Save data
            np.savez(fit_3d_dir, pts_3d = Compiled_landmarks, rot_angles = Compiled_orientation, trans=Compiled_translation)
            np.save(mean_dir, Mean_3d_Position)
            np.save(tracked2D_normalized_pts_fix_contour, Compiled_landmarks_2D)

            remove_folders(frames_dir)

            


            

        

                






