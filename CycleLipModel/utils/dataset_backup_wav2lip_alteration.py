import os
import torch
import pickle
import numpy as np
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
    # assert split in ['test', 'val', 'train']
    # filelist = list(glob(os.path.join(data_root, ('{}/*/*/*/*.jpg').format(split))))
    



    count = 0
    with open('/fs04/za99/scratch_nobackup/jgoh/Preprocessed_VoxCeleb_Lip2Wav/val_info/val_trial.txt') as f:
        for line in f:
            # if count % 100 == 0:
            #     print('Current count is {}'.format(count))
            # count += 1
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(line)
                
    print('{} filelist contains {} videos'.format(split, len(filelist)))

    return filelist


class ljdataset(object):
    def __init__(self, data_root, split):
        self.all_videos = get_image_list(data_root, split)

    def get_frame_id(self, frame):
            return int(basename(frame).split('.')[0])

    def get_lip2wav_window(self, center_frame):
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
    
    def get_wav2lip_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + hps.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames, img_size):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (img_size, img_size))
            except Exception as e:
                return None

            window.append(img)

        return window
    
    def read_window_wav2lip(self, window_fnames, img_size):
        if window_fnames is None: return None
        window = []
        window_set = [] ## This is so that we keep the images as sets of 5
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (img_size, img_size))
            except Exception as e:
                return None

            window_set.append(img)

            if len(window_set) == hps.syncnet_T:
                window_set = np.asarray(window_set)
                window.append(window_set)
                window_set = []

        return window

    # def wav2lip_crop_audio_window(self, spec, start_frame):
    #     # estimate total number of frames from spec (num_features, T)
    #     # num_frames = (T x hop_size * fps) / sample_rate
    #     if type(start_frame) == int:
    #         start_frame_num = start_frame
    #     else:
    #         start_frame_num = self.get_frame_id(start_frame)
    #     # print("The frame num is : {}".format(start_frame_num))
    #     start_idx = int(80. * (start_frame_num / float(hps.fps)))
        
    #     end_idx = start_idx + hps.syncnet_mel_step_size
    #     # print("Wav2lip start idx is : {}".format(start_idx))
    #     # print(end_idx)
    #     # print(spec.shape)

    #     return spec[start_idx : end_idx, :]

    def wav2lip_crop_audio_window_modified(self, spec, start_frame):
        # estimate total number of frames from spec (num_features, T)
        # num_frames = (T x hop_size * fps) / sample_rate
        if type(start_frame) == int:
            start_frame_id = start_frame
        else:
            start_frame_id = self.get_frame_id(start_frame) - hps.T//2
        # print("The frame num is : {}".format(start_frame_num))
        total_num_frames = int((spec.shape[0] * hps.hop_size * hps.fps) / hps.sample_rate)

        start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
        
        end_idx = start_idx + hps.syncnet_mel_step_size
        # print("Wav2lip start idx is : {}".format(start_idx))
        # print(end_idx)
        # print(spec.shape)



        return spec[start_idx : end_idx, :]
    
    
        
    def lip2wav_crop_audio_window(self, spec, center_frame):
        # estimate total number of frames from spec (num_features, T)
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_id = self.get_frame_id(center_frame) - hps.T//2
        total_num_frames = int((spec.shape[0] * hps.hop_size * hps.fps) / hps.sample_rate)

        start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
        # print("lip2wav start idx {}".format(start_idx))
        end_idx = start_idx + hps.mel_step_size

        return spec[start_idx : end_idx, :]
    
    # def get_segmented_mels(self, spec, start_frame):
    #     mels = []
    #     assert hps.syncnet_T == 5
    #     start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
    #     if start_frame_num - 2 < 0: return None
    #     for i in range(start_frame_num, start_frame_num + hps.syncnet_T):
    #         m = self.wav2lip_crop_audio_window(spec, i - 2)
    #         if m.shape[0] != hps.syncnet_mel_step_size:
    #             return None
    #         mels.append(m.T)

    #     mels = np.asarray(mels)
    #     return mels, start_frame_num

    def get_segmented_mels_modified(self, spec, center_frame):
        mels = []
        
        assert hps.syncnet_T == 5
        center_id = self.get_frame_id(center_frame)
        ## so for now lip2wav needs 25 frames but wav2lip needs 5 so we are going to do this in sets
        start_id = center_id - hps.T//2

        # if hps.T%2:
        #     window_ids = range(center_id - hps.T//2, center_id + hps.T//2 + 1 , hps.syncnet_T)
        # else:
        #     window_ids = range(center_id - hps.T//2, center_id + hps.T//2, hps.syncnet_T)


        mels_indiv = []
        if start_id - 2 < 0: return None

        for i in range(start_id, start_id + hps.T):
            m = self.wav2lip_crop_audio_window_modified(spec, i - 2)
            if m.shape[0] != hps.syncnet_mel_step_size:
                return None
            mels_indiv.append(m.T)
            
        mels = np.asarray(mels_indiv)
        
        return mels
    
    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x
    
    def prepare_wav2lip_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (0, 4, 1, 2, 3))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            
            idx = np.random.randint(len(self.all_videos))
            vidname = self.all_videos[idx]
            ## vidname is actual pics already so need to fix here
            img_names = list(glob(os.path.join(vidname, '*/*.jpg')))
            
            if len(img_names) <= 3 * hps.T:
                continue
            
            idx_img = np.random.randint(len(img_names))
            idx_wrong_img = np.random.randint(len(img_names))
            while idx_img == idx_wrong_img:
                idx_wrong_img = np.random.randint(len(img_names))

            img_name = img_names[idx_img]
            wrong_img_name = img_names[idx_wrong_img]

            if not os.path.isfile(os.path.join(os.path.dirname(img_name), 'mels.npz')):
                continue

            if not os.path.isfile(os.path.join(os.path.dirname(img_name), 'ref.npz')):
                continue

            lip2wav_window_fnames = self.get_lip2wav_window(img_name)
            wav2lip_window_fnames = self.get_lip2wav_window(img_name)
            wrong_wav2lip_window_fnames = self.get_lip2wav_window(wrong_img_name)

            if wav2lip_window_fnames is None or wrong_wav2lip_window_fnames is None or lip2wav_window_fnames is None:
                idx = np.random.randint(len(self.all_videos))
                continue

            lip2wav_window = self.read_window(lip2wav_window_fnames, hps.lip2wav_img_size)
            if lip2wav_window is None:
                continue

            wav2lip_window = self.read_window(wav2lip_window_fnames, hps.wav2lip_img_size)
            if wav2lip_window is None:
                continue

            wrong_wav2lip_window = self.read_window(wrong_wav2lip_window_fnames, hps.wav2lip_img_size)
            if wrong_wav2lip_window is None:
                continue

            orig_mel = np.load(os.path.join(os.path.dirname(img_name), 'mels.npz'))['mel'].T
        
            lip2wav_mel = self.lip2wav_crop_audio_window(orig_mel, img_name)
            wav2lip_mel = self.lip2wav_crop_audio_window(orig_mel, img_name)

            if (lip2wav_mel.shape[0] != hps.mel_step_size):
                idx = np.random.randint(len(self.all_videos))
                continue
            indiv_mels = self.get_segmented_mels_modified(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            ## Lip2wav items
            lip2wav_window = self.prepare_window(lip2wav_window)
            lip2wav_x = lip2wav_window.copy()

            ## Wav2lip Items
            wav2lip_window = self.prepare_window(wav2lip_window)
            y = wav2lip_window.copy()
            wav2lip_window[:, :, :, wav2lip_window.shape[2]//2:] = 0.

            
            wrong_wav2lip_window = self.prepare_window(wrong_wav2lip_window)
            wav2lip_x = np.concatenate([wav2lip_window, wrong_wav2lip_window], axis=0) ## moving this so the channel concatenation is in the correct position
    	    
            ## speaker embedding
            refs = np.load(os.path.join(os.path.dirname(img_name), 'ref.npz'))['ref']
            ref = refs[np.random.choice(len(refs))]

            ## final wav2lip data changes
            wav2lip_mel = wav2lip_mel.T
            
            return lip2wav_x, lip2wav_mel, ref , wav2lip_x, indiv_mels, wav2lip_mel, y
    
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
            ## Colation of lip2wav
            batch = batch[size_per_device*i:size_per_device*(i+1)]
            input_cur_device, input_max_len = self._prepare_inputs([x[0] for x in batch])
            inputs = torch.FloatTensor(np.concatenate((inputs, input_cur_device), axis=1) if inputs is not None else input_cur_device)
            mel_target_cur_device, mel_target_max_len = self._prepare_targets([x[1] for x in batch], hps.outputs_per_step)
            mel_targets = torch.FloatTensor(np.concatenate(( mel_targets, mel_target_cur_device), axis=1) if mel_targets is not None else mel_target_cur_device)

            # ## Collation of wav2lip
            # input_wav2lip_device = self._prepare_wav2lip_inputs([x[3] for x in batch])
            # wav2lip_inputs = torch.FloatTensor(np.concatenate((wav2lip_inputs, input_wav2lip_device), axis=1) if wav2lip_inputs is not None else input_wav2lip_device)
            # indiv_mels_target_cur_device = self._prepare_wav2lip_targets([x[4] for x in batch], hps.outputs_per_step)
            # indiv_mels = torch.FloatTensor(np.concatenate(( indiv_mels, indiv_mels_target_cur_device), axis=1) if indiv_mels is not None else indiv_mels_target_cur_device)
            # window_target_cur_device = self._prepare_wav2lip_targets([x[5] for x in batch], hps.outputs_per_step)
            # y_window = torch.FloatTensor(np.concatenate(( y_window, window_target_cur_device), axis=1) if y_window is not None else window_target_cur_device)

            #Pad sequences with 1 to infer that the sequence is done
            #token_target_cur_device, token_target_max_len = self._prepare_token_targets([x[2] for x in batch], outputs_per_step)
            #token_targets = np.concatenate((token_targets, token_target_cur_device),axis=1) if token_targets is not None else token_target_cur_device
            split_infos.append([input_max_len, mel_target_max_len])

        split_infos = torch.IntTensor(np.asarray(split_infos, dtype=np.int32))
        
        ### SV2TTS ###
        
        #embed_targets = np.asarray([x[3] for x in batches])
        embed_targets = torch.FloatTensor(np.asarray([x[2] for x in batch]))

        ##No collation required for wav2lip so leave it as such
        wav2lip_inputs = torch.FloatTensor(np.asarray([x[3] for x in batch]))
        indiv_mels = torch.FloatTensor(np.asarray([x[4] for x in batch])).unsqueeze(2)
        wav2lip_mel = torch.FloatTensor(np.asarray([x[5] for x in batch])).unsqueeze(1)
        y_window = torch.FloatTensor(np.asarray([x[6] for x in batch]))


        
        return inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets  \
        , wav2lip_inputs, indiv_mels, wav2lip_mel, y_window
    
    def _prepare_inputs(self, inputs):
        max_len = max([x.shape[1] for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len
    
    def _prepare_wav2lip_inputs(self, inputs):
        max_len = max([x.shape[1] for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs])

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len
    
    def _prepare_wav2lip_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets])
    
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
    


