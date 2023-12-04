import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import cv2

from .lip2wav import Tacotron2Loss, Tacotron2 as lip2wav
from .wav2lip import Wav2Lip, Wav2Lip_disc_qual
from hparams import hparams


class cyclelip(nn.Module):
    def __init__(self, device, n_gpu, lip2wav, wav2lip, disc, lip2wav_optim, wav2lip_optim, disc_optim, tacotron_loss, sync_loss):
        # These are the models we will be working with
        super(cyclelip,self).__init__()
        self.device = device

        self.lip2wav = lip2wav
        self.lip2wav_optim = lip2wav_optim
        self.tacotron_loss= Tacotron2Loss()

        self.wav2lip = wav2lip
        self.wav2lip_optim = wav2lip_optim
        self.get_sync_loss = sync_loss
        self.recon_loss = nn.L1Loss()

        self.qual_disc = disc
        self.disc_optim = disc_optim

        self.n_gpu = n_gpu

        self.cycle_criterion = nn.L1Loss()


        #L

    def forward(self, lip2wav_batch, wav2lip_batch):
        frames_for_lip2wav = []
        five_frame_list = []
        frame_reshape = []

        self.lip2wav_x, self.lip2wav_y = (self.lip2wav.module if self.n_gpu > 1 else self.lip2wav).parse_batch(lip2wav_batch)
        wav2lip_x, indiv_mels, wav2lip_mel, self.wav2lip_y = wav2lip_batch

        print(wav2lip_x.shape)
        print(indiv_mels.shape)
        ## Training step for training (This will be used to update our weights)
        ## Cycle A
        self.lip2wav_out, self.trainable_params_lip2wav = self.lip2wav(self.lip2wav_x)
        self.wav2lip_out = self.wav2lip(indiv_mels, wav2lip_x)

        ## Inference step to prepare for next cycle (This will be used for the next cycle)
        # self.wav2lip.eval()
        # self.lip2wav.eval()

        # for mel_ind, face_x in zip(indiv_mels, wav2lip_x):            
            


        #     # Disable gradient computation
        #     with torch.no_grad():  
        #         wav2lip_frames = self.wav2lip(mel_ind, face_x)

        #         for frames in wav2lip_frames:
        #             # print(frames.shape)
        #             list_of_frames = frames.permute(1, 0, 2, 3) * 255.

        #             for frame in list_of_frames:

        #                 reshape_frame = F.interpolate(frame.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
            
        #                 frame_reshape.append(reshape_frame)
                    
        #             frame_reshape = torch.cat(frame_reshape, dim = 0)
        #             frame_reshape = frame_reshape.unsqueeze(0)

        #             five_frame_list.append(frame_reshape)
        #             frame_reshape= []
                
                
        #         lip2wav_tensor = torch.cat(five_frame_list, dim = 0)
        #         frames_for_lip2wav.append(lip2wav_tensor)
        #         five_frame_list = []

        # frames_for_lip2wav = torch.cat(frames_for_lip2wav, dim = 1)
        # frames_for_lip2wav = frames_for_lip2wav.permute(0, 2, 1, 3, 4)        


        # print("Original  frames shape is {}".format(self.lip2wav_x[0].shape))
        # print("new Wav2lip frames shape is {}".format(frames_for_lip2wav.shape))

        # self.lip2wav_audio = self.lip2wav_out[0]

        # lip2wav_indiv_mel = self.lip2wav_audio.reshape()
        # ## End inference step and restart training process
        # self.wav2lip.train()
        # self.lip2wav.train()

        ## Cycle B begins
        print("Original indiv_mels audio shape is {}".format(indiv_mels.shape))
        print("new lip2wav audio shape is {}".format(self.lip2wav_audio.shape))

        
        ##Insert new frames and mels for the next data 

        new_lip2wav_x = (frames_for_lip2wav, self.lip2wav_x[1], self.lip2wav_x[2], self.lip2wav_x[3], \
            self.lip2wav_x[4], self.lip2wav_x[5])

        ##Some processing magic goes here so that we change what are the inputs
        self.synced_lip2wav_audio, self.trainable_params_lip2wav = self.lip2wav(new_lip2wav_x)
        self.lipread_wav2lip_frames = self.wav2lip(lip2wav_indiv_mel, wav2lip_x[0])
            
        return self.lip2wav_audio, self.wav2lip_frames, self.synced_lip2wav_audio, self.lipread_wav2lip_frames, self.trainable_params_lip2wav

    def zero_grad_model_wav_models(self):
        self.wav2lip_optim.zero_grad()
        self.lip2wav_optim.zero_grad()
        
    def zero_grad_model_disc_model(self):
        self.disc_optim.zero_grad()
    
    def optimize_wav_models(self, lip2wav_audio, wav2lip_frames, synced_lip2wav_audio, lipread_wav2lip_frames, mel, gt, trainable_params):
        lip2wav_loss = self.get_lip2wav_loss(lip2wav_audio, mel, trainable_params)
        cycleLoss_A = self.cycle_criterion(lip2wav_audio, synced_lip2wav_audio)

        wav2lip_loss = self.get_wav2lip_loss(mel, wav2lip_frames, gt)
        cycleLoss_B = self.cycle_criterion(wav2lip_frames, lipread_wav2lip_frames)

        total_loss = lip2wav_loss + cycleLoss_A + wav2lip_loss + cycleLoss_B

        total_loss.backward()

        grad_norm_lip2wav = torch.nn.utils.clip_grad_norm_(self.lip2wav.parameters(), hparams.tacotron_clip_gradients_clip_thresh)
        grad_norm_wav2lip = torch.nn.utils.clip_grad_norm_(self.wav2lip.parameters(), hparams.tacotron_clip_gradients_clip_thresh)

        self.wav2lip_optim.step()
        self.lip2wav_optim.step()

        return lip2wav_loss, cycleLoss_A, wav2lip_loss, cycleLoss_B, grad_norm_lip2wav, grad_norm_wav2lip

    
    def optimize_disc_model(self, gt, g_original, g_lip2wav):
        pred = self.qual_disc(gt)
        disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(self.device))
        disc_real_loss.backward()

        pred_original = self.qual_disc(g_original.detach())
        pred_lip2wav = self.qual_disc(g_lip2wav.detach()) 
        disc_fake_loss_ori = F.binary_cross_entropy(pred_original, torch.zeros((len(pred_original), 1)).to(self.device))
        disc_fake_loss_lip2wav = F.binary_cross_entropy(pred_lip2wav, torch.zeros((len(pred_lip2wav), 1)).to(self.device))
        total_disc_fake_loss = disc_fake_loss_ori + disc_fake_loss_lip2wav
        total_disc_fake_loss.backward()

        self.disc_optim.step()

        return disc_real_loss, total_disc_fake_loss


        

    def get_lip2wav_loss(self, y_pred, y, trainable_params):
        loss, items = self.tacotron_loss(y_pred, y, trainable_params)

        return loss

    def get_wav2lip_loss(self, mel, g, gt):
        if hparams.syncnet_wt > 0.:
                sync_loss = self.get_sync_loss(mel, g)
        else:
            sync_loss = 0.

        if hparams.disc_wt > 0.:
            perceptual_loss = self.qual_disc.perceptual_forward(g)
        else:
            perceptual_loss = 0.
        
        l1loss = self.recon_loss(g, gt)

        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss
        return loss


### Data processing for new cycle consistency

## wav2lip data prep
    def get_segmented_mels(self, spec, start_frame):
        mels = []
        batch_mels = []
        start_frame_num = start_frame # 0-indexing ---> 1-indexing
        for x in spec:
            for i in range(start_frame_num, start_frame_num + hparams.syncnet_T):
                m = self.crop_audio_window(x, i - 2)
                if m.shape[0] != hparams.mel_step_size:
                    return None
                mels.append(m.T)
            mels = np.asarray(mels)
            batch_mels.append(mels)
        
        batch_mels = np.asarray(batch_mels)

        return batch_mels
    
    def crop_audio_window(self, spec, center_frame):
        # estimate total number of frames from spec (num_features, T)
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_id = self.get_frame_id(center_frame) - hparams.T//2
        total_num_frames = int((spec.shape[0] * hparams.hop_size * hparams.fps) / hparams.sample_rate)

        start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
        end_idx = start_idx + hparams.mel_step_size

        return spec[start_idx : end_idx, :]
