import os
import torch
import pickle
import numpy as np
from text import text_to_sequence
from hparams import hparams as hps
from torch.utils.data import Dataset
from utils.audio import load_wav, melspectrogram

## addition by Jared
from os.path import dirname, join, basename, isfile
from glob import glob
import os, random, cv2, argparse
from itertools import islice
from hparams import hparams as hps

#For the purpose of testing
import sys


def get_image_list(data_root, split):
    # #Samller dataset for testing purposes
    filelist = []
    assert split in ['test', 'val', 'train']
    filelist = list(glob(os.path.join(data_root, ('{}/*/*/*/*.jpg').format(split))))
    



    count = 0
    # with open('/fs04/za99/scratch_nobackup/jgoh/Preprocessed_VoxCeleb_Lip2Wav/val_info/val_trial.txt') as f:
    #     for line in f:
    #         # if count % 100 == 0:
    #         #     print('Current count is {}'.format(count))
    #         # count += 1
    #         line = line.strip()
    #         if ' ' in line: line = line.split()[0]
    #         for file in glob(os.path.join(line, '*/*.jpg')):
    #             filelist.append(file)
    print('{} filelist contains {} images'.format(split, len(filelist)))

    return filelist


class ljdataset(Dataset):
    def __init__(self, data_root, split):
        self.all_videos = get_image_list(data_root, split)

    def get_frame_id(self, frame):
            return int(basename(frame).split('.')[0])

    def get_window(self, center_frame):
        center_id = self.get_frame_id(center_frame)
        vidname = dirname(center_frame)
        if hps.T%2:
            window_ids = range(center_id - hps.T//2, center_id + hps.T//2 + 1)
        else:
            window_ids = range(center_id - hps.T//2, center_id + hps.T//2)

        window_fnames = []
        for frame_id in window_ids:
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hps.img_size, hps.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, center_frame):
        # estimate total number of frames from spec (num_features, T)
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_id = self.get_frame_id(center_frame) - hps.T//2
        total_num_frames = int((spec.shape[0] * hps.hop_size * hps.fps) / hps.sample_rate)

        start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
        end_idx = start_idx + hps.mel_step_size

        return spec[start_idx : end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = np.random.randint(len(self.all_videos))

            img_name = self.all_videos[idx]

            if not os.path.isfile(os.path.join(os.path.dirname(img_name), 'mels.npz')):
                continue

            if not os.path.isfile(os.path.join(os.path.dirname(img_name), 'ref.npz')):
                continue

            window_fnames = self.get_window(img_name)
            if window_fnames is None:
                idx = np.random.randint(len(self.all_videos))
                continue

            mel = np.load(os.path.join(os.path.dirname(img_name), 'mels.npz'))['mel'].T
            mel = self.crop_audio_window(mel, img_name)
            if (mel.shape[0] != hps.mel_step_size):
                idx = np.random.randint(len(self.all_videos))
                continue
            break
        # print("shape of raw mel is {}".format(mel.shape))

        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            try:
                img = cv2.resize(img, (hps.img_size, hps.img_size))
            except:
                continue

            window.append(img)
        x = np.asarray(window) / 255.
        x = x.transpose(3, 0, 1, 2) # For pytorch implementation channel must be last

        ## speaker embedding
        refs = np.load(os.path.join(os.path.dirname(img_name), 'ref.npz'))['ref']
        ref = refs[np.random.choice(len(refs))]
        
        return x, mel, ref

# class ljcollate():
#     def __init__(self, n_frames_per_step):
#         self.n_frames_per_step = n_frames_per_step

#     def __call__(self, batch):
#         # Right zero-pad all one-hot text sequences to max input length
#         # input_lengths, ids_sorted_decreasing = torch.sort(
#         #     torch.LongTensor([len(x[0]) for x in batch]),
#         #     dim=0, descending=True)
#         # max_input_len = input_lengths[0]

#         # text_padded = torch.LongTensor(len(batch), max_input_len)
#         # text_padded.zero_()
#         # for i in range(len(ids_sorted_decreasing)):
#         #     text = batch[ids_sorted_decreasing[i]][0]
#         #     text_padded[i, :text.size(0)] = text
        
#         # Right zero-pad Frames
#         input_lengths, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([len(x[0]) for x in batch]),
#             dim=0, descending=True)

#         num_frames =len(batch[0][0])
#         max_frames_len = max([len(x[0]) for x in batch])
#         if max_frames_len % self.n_frames_per_step != 0:
#             max_frames_len += self.n_frames_per_step - max_frames_len % self.n_frames_per_step
#             assert max_frames_len % self.n_frames_per_step == 0

#         frames_padded = torch.FloatTensor(len(batch), max_frames_len, batch[0][0].shape[3], batch[0][0].shape[2], batch[0][0].shape[1])
#         frames_padded.zero_()

#         # Right zero-pad mel-spec
#         num_mels = len(batch[0][1])
#         max_target_len = max([len(x[1]) for x in batch])
#         if max_target_len % self.n_frames_per_step != 0:
#             max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
#             assert max_target_len % self.n_frames_per_step == 0

#         # include mel padded and gate padded
#         mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
#         mel_padded.zero_()
#         gate_padded = torch.FloatTensor(len(batch), max_target_len)
#         gate_padded.zero_()
#         output_lengths = torch.LongTensor(len(batch))
        
#         ## Jared Ref Padding for collation
#         num_ref = len(batch[0][2])
#         max_ref_length = max([len(x[2]) for x in batch])
#         ref_padded = torch.FloatTensor(len(batch), max_ref_length)
#         ref_padded.zero_()

#         for i in range(len(ids_sorted_decreasing)):
#             mel = batch[ids_sorted_decreasing[i]][1]
#             mel_tensor = torch.from_numpy(mel) #jared addition for conversion of numpy to torch tensor

#             mel_padded[i, :, :num_mels] = mel_tensor
#             gate_padded[i, num_mels-1:] = 1
#             output_lengths[i] = num_mels

#             ## Jared addition

#             frames = batch[ids_sorted_decreasing[i]][0]
#             frames_tensor = torch.from_numpy(frames) #jared addition for conversion of numpy to torch tensor
#             frames_padded[i, :num_frames, :, :, :] = torch.transpose(frames_tensor, 3, 1)

#             ref = batch[ids_sorted_decreasing[i]][2]
#             ref_tensor = torch.from_numpy(ref) #jared addition for conversion of numpy to torch tensor
#             ref_padded[i, :num_ref] = ref_tensor

#         return frames_padded, input_lengths, mel_padded, ref_padded ,gate_padded, output_lengths
    
class ljcollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step
        self._pad = 0
        self._token_pad = 1
        if hps.symmetric_mels:
            self._target_pad = -hps.max_abs_value
        else:
            self._target_pad = 0.

    def __call__(self, batch):
        size_per_device = int(len(batch) / hps.tacotron_num_gpus)
        np.random.shuffle(batch)

        inputs = None
        mel_targets = None
        #token_targets = None
        targets_lengths = None
        split_infos = []

        targets_lengths = torch.IntTensor(np.asarray([len(x[1]) for x in batch], dtype=np.int32)) #Used to mask loss
        input_lengths = torch.IntTensor(np.asarray([x[0].shape[1] for x in batch], dtype=np.int32))

        for i in range(hps.tacotron_num_gpus):
            batch = batch[size_per_device*i:size_per_device*(i+1)]
            input_cur_device, input_max_len = self._prepare_inputs([x[0] for x in batch])
            inputs = torch.FloatTensor(np.concatenate((inputs, input_cur_device), axis=1) if inputs is not None else input_cur_device)
            mel_target_cur_device, mel_target_max_len = self._prepare_targets([x[1] for x in batch], hps.outputs_per_step)
            mel_targets = torch.FloatTensor(np.concatenate(( mel_targets, mel_target_cur_device), axis=1) if mel_targets is not None else mel_target_cur_device)

            #Pad sequences with 1 to infer that the sequence is done
            #token_target_cur_device, token_target_max_len = self._prepare_token_targets([x[2] for x in batch], outputs_per_step)
            #token_targets = np.concatenate((token_targets, token_target_cur_device),axis=1) if token_targets is not None else token_target_cur_device
            split_infos.append([input_max_len, mel_target_max_len])

        split_infos = torch.IntTensor(np.asarray(split_infos, dtype=np.int32))
        
        ### SV2TTS ###
        
        #embed_targets = np.asarray([x[3] for x in batches])
        embed_targets = torch.FloatTensor(np.asarray([x[2] for x in batch]))

        ##############
        
        #return inputs, input_lengths, mel_targets, token_targets, targets_lengths, \
        #	   split_infos, embed_targets


        
        return inputs, input_lengths, mel_targets, targets_lengths, \
            split_infos, embed_targets
    
    def _prepare_inputs(self, inputs):
        max_len = max([x.shape[1] for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len
    
    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[1]), mode="constant", constant_values=self._pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode="constant", constant_values=self._target_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode="constant", constant_values=self._token_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder
    


