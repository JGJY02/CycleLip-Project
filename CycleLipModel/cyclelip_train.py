from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual, Wav2LipLoss
from models import Tacotron2 as Lip2Wav
from models import cyclelip
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams
from utils.dataset import ljdataset, ljcollate
from torch.utils.data import DistributedSampler, DataLoader
from models import Tacotron2, Tacotron2Loss

##Plot lip2wav results
from utils import plot
from utils import audio



parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument("--data_root", default = "/fs04/za99/scratch_nobackup/jgoh/Preprocessed_VoxCeleb_Lip2Wav", help="Root folder of the preprocessed LRS2 dataset", required=False, type=str)

parser.add_argument('--checkpoint_dir_wav2lip', help='Save checkpoints to this directory', required=False, type=str, default = '/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.2-weight-model/wav2lip')
parser.add_argument('--syncnet_checkpoint_path', default = '/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/wav2lip-models/Discriminator_Models/prev_models/checkpoint_step000070000.pth', help='Load the pre-trained Expert discriminator', required=False, type=str)

parser.add_argument('--checkpoint_path_wav2lip', help='Resume generator from this checkpoint', default='/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/wav2lip-models/Wav2lip_model/checkpoint_step000000050.pth', type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default='/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/wav2lip-models/Wav2lip_model/disc_checkpoint_step000000050.pth', type=str)

## 1000 model path discriminator /fs04/za99/jgoh/Wav2Lip-master/training_results/models/disc_checkpoint_step000001000.pth 
## 1000 model path wav2lip /fs04/za99/jgoh/Wav2Lip-master/training_results/models/checkpoint_step000001000.pth

## 50 model path discriminator /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/wav2lip-models/Wav2lip_model/disc_checkpoint_step000000050.pth 
## 50 model path wav2lip /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/wav2lip-models/Wav2lip_model/checkpoint_step000000050.pth


## Lip2wav paths
parser.add_argument('--checkpoint_dir_lip2wav', help='Save lip2wav model to this directory', default = '/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.2-weight-model/lip2wav', required=False, type=str)
parser.add_argument('--checkpoint_path_lip2wav', help='Load the warmedup lip2wav', default = '/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/lip2wav-pytorch/ckpt_50', required=False, type=str)

## 1000 model path /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Lip2Wav-pytorchConversion/LIP2WAV-Converted/training_results/model/ckpt_1000

## 50 model path /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/lip2wav-pytorch/ckpt_50

## remember to change this as well
parser.add_argument('--checkpoint_dir_loss_data', help='this is where the loss data is stored', default = '/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.2-weight-model/loss_data')
## Training Params
parser.add_argument("--train_steps", type=int, default=2000000, # Was 100000
                        help="total number of tacotron training steps")
parser.add_argument("--checkpoint_interval", type=int, default=10000, # Was 10000
                    help="Steps between writing checkpoints")
parser.add_argument("--eval_interval", type=int, default=1000, # Was 10000
                    help="Steps between eval on test data")
parser.add_argument("--data_log_interval", type=int, default=5000, # Was 5000
                    help="Steps between writing checkpoints")
parser.add_argument("--cycle_weight", type=float, default=1, # Was 5000
                    help="weight of cycle consistency")


args = parser.parse_args()

class ExponentialLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr, tacotron_start_decay, decay_steps, decay_rate,
                 warmup_steps, last_epoch=-1):
        self.init_lr = init_lr
        self.tacotron_start_decay = tacotron_start_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        super(ExponentialLRWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.tacotron_start_decay:
            lr = self.init_lr
        elif self.init_lr > hparams.tacotron_final_learning_rate:
            decay_steps = self.decay_steps
            if self.last_epoch < self.tacotron_start_decay + self.warmup_steps:
                decay_steps -= (self.tacotron_start_decay + self.warmup_steps - self.last_epoch)
            lr = self.init_lr * (self.decay_rate ** (self.last_epoch // decay_steps))
        else:
            lr = hparams.tacotron_final_learning_rate

        return [lr]
    
def prepare_dataloaders(split):
    print("dataset preperation has begun")
    trainset = ljdataset(args.data_root, split)
    collate_fn = ljcollate(hparams.outputs_per_step)
    # sampler = DistributedSampler(trainset) if n_gpu > 1 else None
    if split == 'train':
        train_loader = DataLoader(trainset, num_workers = 16, shuffle = True,
                                batch_size = hparams.batch_size, collate_fn = collate_fn)
    else:
        train_loader = DataLoader(trainset, num_workers = 4, shuffle = True,
                                 batch_size = hparams.batch_size, collate_fn = collate_fn)
    
    return train_loader


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

def eval_model_lip2wav(test_data_loader, criterion, n_gpu, model):
    
    eval_steps = 300
    cur_step = 0
    print('\n Evaluating lip2wav for {} steps \n'.format(eval_steps))
    total_loss, before_loss, after_loss, l1_loss, regularization_loss = [], [], [], [], []

    for batch in test_data_loader:
        model.eval()
        inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets, _, _, _, _ = batch
        lip2wav_batch = (inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets)

        x, y = (model.module if n_gpu > 1 else model).parse_batch(lip2wav_batch)
        y_pred, trainable_params = model(x)

        # loss
        loss, items = criterion(y_pred, y, trainable_params)

        total_loss.append(loss)
        before_loss.append(items[0])
        after_loss.append(items[1])
        l1_loss.append(items[2])
        regularization_loss.append(items[3])

        cur_step += 1
        if cur_step > eval_steps: break
        
    
    avg_total_loss = sum(total_loss) / len(total_loss)
    avg_before_loss = sum(before_loss) / len(before_loss)
    avg_after_loss = sum(after_loss) / len(after_loss)
    avg_l1_loss = sum(l1_loss) / len(l1_loss)
    avg_regularization_loss = sum(regularization_loss) / len(regularization_loss)

    validation_losses = [avg_total_loss, avg_before_loss, avg_after_loss, avg_l1_loss, avg_regularization_loss]


    print('Lip2wav Evaluation Results | Total loss: {} | before Postnet: {}, after Postnet: {}, l1: {} Regularization: {}'.format(sum(total_loss) / len(total_loss), sum(before_loss) / len(before_loss),
                                                        sum(after_loss) / len(after_loss),
                                                        sum(l1_loss) / len(l1_loss),
                                                        sum(regularization_loss) / len(regularization_loss)))
    
    return validation_losses

def cosine_loss(a, v, y):
    logloss = nn.BCELoss()
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(hparams.syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def eval_model_wav2lip(test_data_loader, device, model, disc):
    eval_steps = 300
    print('\n Evaluating wav2lip for {} steps \n'.format(eval_steps))
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss, total_loss = [], [], [], [], [], []
    while 1:
        for step, (_, _, _, _, _, _, x, indiv_mels, mel, gt) in enumerate((test_data_loader)):
            model.eval()
            disc.eval()

            

            x = x[0].to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels[0].to(device)
            gt = gt[0].to(device)

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            g = model(indiv_mels, x)
            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = get_sync_loss(mel, g)
            
            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss
            total_loss.append(loss.item())

            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            
            if hparams.disc_wt > 0.:
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            if step > eval_steps: break

        print('Wav2lip Evaluation Results | Total Loss: {} | L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(sum(total_loss) / len(total_loss), 
                                                                                                                       sum(running_l1_loss) / len(running_l1_loss),
                                                            sum(running_sync_loss) / len(running_sync_loss),
                                                            sum(running_perceptual_loss) / len(running_perceptual_loss),
                                                            sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                                                            sum(running_disc_real_loss) / len(running_disc_real_loss)))
        
        avg_total_loss = sum(total_loss) / len(total_loss)
        avg_l1_loss = sum(running_l1_loss) / len(running_l1_loss)
        avg_sync_loss = sum(running_sync_loss) / len(running_sync_loss)
        avg_percep_loss = sum(running_perceptual_loss) / len(running_perceptual_loss)
        avg_fake_loss = sum(running_disc_fake_loss) / len(running_disc_fake_loss)
        avg_real_loss = sum(running_disc_real_loss) / len(running_disc_real_loss)

        validation_losses = [avg_total_loss, avg_sync_loss, avg_percep_loss, avg_l1_loss, avg_real_loss, avg_fake_loss]


        return validation_losses
        
        

def train(device, cyclemodel, train_data_loader, test_data_loader,
          checkpoint_dir_wav2lip=None, checkpoint_dir_lip2wav=None, checkpoint_interval=None, nepochs=None, data_log_interval=None, hparams = hparams):
    global global_step, global_epoch

    plot_dir = os.path.join(args.checkpoint_dir_loss_data, "plots")
    wav_dir = os.path.join(args.checkpoint_dir_loss_data, "wavs")
    mel_dir = os.path.join(args.checkpoint_dir_loss_data, "mel-spectrograms")

    lip2wav_loss_dir = os.path.join(args.checkpoint_dir_loss_data, "lip2wav")
    lip2wav_loss_train_dir = os.path.join(args.checkpoint_dir_loss_data, "lip2wav", "train")
    lip2wav_loss_val_dir = os.path.join(args.checkpoint_dir_loss_data, "lip2wav", "val")

    wav2lip_loss_dir = os.path.join(args.checkpoint_dir_loss_data, "wav2lip")
    wav2lip_loss_train_dir = os.path.join(args.checkpoint_dir_loss_data, "wav2lip", "train")
    wav2lip_loss_val_dir = os.path.join(args.checkpoint_dir_loss_data, "wav2lip", "val")

    cycle_loss_dir = os.path.join(args.checkpoint_dir_loss_data, "cycle")

    lip2wav_dir = args.checkpoint_dir_lip2wav
    wav2lip_dir = args.checkpoint_dir_wav2lip

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(lip2wav_dir, exist_ok=True)
    os.makedirs(wav2lip_dir, exist_ok=True)

    os.makedirs(lip2wav_loss_dir, exist_ok=True)
    os.makedirs(lip2wav_loss_train_dir, exist_ok=True)
    os.makedirs(lip2wav_loss_val_dir, exist_ok=True)
    
    os.makedirs(wav2lip_loss_dir, exist_ok=True)
    os.makedirs(wav2lip_loss_train_dir, exist_ok=True)
    os.makedirs(wav2lip_loss_val_dir, exist_ok=True)

    os.makedirs(cycle_loss_dir, exist_ok = True)

    resumed_step = global_step

    lip2wav_loss_array, before_postnet_loss_array, after_postnet_loss_array, lip2wav_l1_loss_array, regularization_array = [], [], [], [], []
    wav2lip_loss_array, wav2lip_l1_loss_array, sync_loss_array, percep_loss_array, fake_loss_array, real_loss_array = [], [] ,[], [] ,[] ,[]
    cycle_loss_A_array, cycle_loss_B_array= [], []

    avg_val_lip2wav_loss_array, avg_val_before_postnet_loss_array, avg_val_after_postnet_loss_array, avg_val_lip2wav_l1_loss_array, avg_val_regularization_array = [], [], [], [], []
    avg_val_wav2lip_loss_array, avg_val_wav2lip_l1_loss_array, avg_val_sync_loss_array, avg_val_percep_loss_array, avg_val_fake_loss_array, avg_val_real_loss_array = [], [] ,[], [] ,[] ,[]


    if hparams.tacotron_decay_learning_rate:
        # lr_lambda = lambda step: hps.tacotron_decay_steps**0.5*min((step+1)*hps.tacotron_decay_steps**-1.5, (step+1)**-0.5)
        # if args.checkpoint_path != '':
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
        # else:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler_lip2wav = ExponentialLRWithWarmup(cyclemodel.lip2wav_optim, hparams.tacotron_initial_learning_rate, hparams.tacotron_start_decay,
                                    hparams.tacotron_decay_steps, hparams.tacotron_decay_rate, 0)
    

    while global_epoch < args.train_steps:
        print('Starting Epoch: {}'.format(global_epoch))
        running_lip2wav_loss, running_wav2lip_loss, running_cycle_loss_A, running_cycle_loss_B = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        
        ## average arrays
        avg_lip2wav_loss_array, avg_before_postnet_loss_array, avg_after_postnet_loss_array, avg_lip2wav_l1_loss_array, avg_regularization_array = [], [], [], [], []
        avg_wav2lip_loss_array, avg_wav2lip_l1_loss_array, avg_sync_loss_array, avg_percep_loss_array, avg_fake_loss_array, avg_real_loss_array = [], [] ,[], [] ,[] ,[]

 

        prog_bar = tqdm(enumerate(train_data_loader))

        for step, batch in prog_bar:
            cyclemodel.train()

            ### Train generator now. Remove ALL grads. 
            cyclemodel.zero_grad_model_wav_models()
            
            inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets, wav2lip_inputs, indiv_mels, wav2lip_mels, y_window = batch
            
            ## Wav2lip data process
            wav2lip_x = wav2lip_inputs.to(device)
            wav2lip_mels = wav2lip_mels.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = y_window.to(device)

            ## Package lip2wav batch
            lip2wav_batch = (inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets)
            wav2lip_batch = (wav2lip_x, indiv_mels, wav2lip_mels, gt)
            #Cyclemodel forward
            lip2wav_audio, wav2lip_frames, synced_lip2wav_audio, lipread_wav2lip_frames, trainable_params_lip2wav, lip2wav_y = cyclemodel(lip2wav_batch, wav2lip_batch)
            ## Get loss and Optimize Models
            lip2wav_loss, lip2wav_indiv_losses, wav2lip_loss, wav2lip_indiv_losess, cycleLoss_A, cycleLoss_B, grad_norm_lip2wav, grad_norm_wav2lip = cyclemodel.optimize_wav_models(lip2wav_audio, wav2lip_frames, synced_lip2wav_audio, lipread_wav2lip_frames,wav2lip_mels ,gt , trainable_params_lip2wav, args.cycle_weight)

            ### Remove all gradients before Training disc
            cyclemodel.zero_grad_model_disc_model()

            disc_real_loss, disc_fake_loss = cyclemodel.optimize_disc_model(gt, wav2lip_frames, lipread_wav2lip_frames)

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()


            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_lip2wav_loss += lip2wav_loss.item()
            running_wav2lip_loss += wav2lip_loss.item()
            running_cycle_loss_A += cycleLoss_A.item()
            running_cycle_loss_B += cycleLoss_B.item()

            ## append losses for figure visualization later
            lip2wav_loss_array.append(lip2wav_loss.item())
            before_postnet_loss_array.append(lip2wav_indiv_losses[0])
            after_postnet_loss_array.append(lip2wav_indiv_losses[1])
            lip2wav_l1_loss_array.append(lip2wav_indiv_losses[2])
            regularization_array.append(lip2wav_indiv_losses[3])

            ##Wav2lip lossess
            wav2lip_loss_array.append(wav2lip_loss.item())
            fake_loss_array.append(disc_fake_loss.item())
            real_loss_array.append(disc_real_loss.item())
            sync_loss_array.append(wav2lip_indiv_losess[0])
            percep_loss_array.append(wav2lip_indiv_losess[1])
            wav2lip_l1_loss_array.append(wav2lip_indiv_losess[2])
            
            ## Cycle Loss
            cycle_loss_A_array.append(cycleLoss_A.item())
            cycle_loss_B_array.append(cycleLoss_B.item())

            if global_step % args.eval_interval == 0:
                with torch.no_grad():
                    lip2wav_val_losses = eval_model_lip2wav(test_data_loader, cyclemodel.tacotron_loss, n_gpu, cyclemodel.lip2wav)
                    avg_val_lip2wav_loss_array.append(lip2wav_val_losses[0])
                    avg_val_before_postnet_loss_array.append(lip2wav_val_losses[1])
                    avg_val_after_postnet_loss_array.append(lip2wav_val_losses[2])
                    avg_val_lip2wav_l1_loss_array.append(lip2wav_val_losses[3])
                    avg_val_regularization_array.append(lip2wav_val_losses[4])

                    wav2lip_val_losses = eval_model_wav2lip(test_data_loader, device, cyclemodel.wav2lip, cyclemodel.qual_disc)
                    avg_val_wav2lip_loss_array.append(wav2lip_val_losses[0])
                    avg_val_sync_loss_array.append(wav2lip_val_losses[1])
                    avg_val_percep_loss_array.append(wav2lip_val_losses[2])
                    avg_val_wav2lip_l1_loss_array.append(wav2lip_val_losses[3])
                    avg_val_real_loss_array.append(wav2lip_val_losses[4])
                    avg_val_fake_loss_array.append(wav2lip_val_losses[5])
                
            if global_step == 1 or global_step % checkpoint_interval == 0:
                print("Saving models ...")
                save_checkpoint_lip2wav(cyclemodel.lip2wav, cyclemodel.lip2wav_optim, global_step, n_gpu)
                save_checkpoint(cyclemodel.qual_disc, cyclemodel.disc_optim, global_step, checkpoint_dir_wav2lip, global_epoch, prefix='disc_')
                save_checkpoint(cyclemodel.wav2lip, cyclemodel.wav2lip_optim, global_step, checkpoint_dir_wav2lip, global_epoch, prefix='wav2lip_')
                
                

                
                print("Models Saved !")

            if global_step == 1 or global_step % data_log_interval == 0:

                ## Wav2lip Pred Save ( Control )
                save_sample_images(wav2lip_x[0], wav2lip_frames, gt[0], global_step, checkpoint_dir_wav2lip)
                
                ## lip2wav Pred Save ( Control )
                # save predicted mel spectrogram to disk (debug)
                step = global_step
                mel_prediction = lip2wav_audio[1][0].cpu().detach().numpy()
                hparams = hparams
                alignment = lip2wav_audio[2][0].cpu().detach().numpy()
                target_length = lip2wav_y[2][0].cpu().detach().numpy()
                target = lip2wav_y[0][0].cpu().detach().numpy()

                mel_filename = "mel-prediction-step-{}.npy".format(step)
                np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T,
                        allow_pickle=False)
                
                # save griffin lim inverted wav for debug (mel -> wav)
                wav = audio.inv_mel_spectrogram(mel_prediction.T)
                audio.save_wav(wav,
                                os.path.join(wav_dir, "step-{}-wave-from-mel.wav".format(step)),
                                sr=hparams.sample_rate)
                
                # save alignment plot to disk (control purposes)
                plot.plot_alignment(alignment,
                                    os.path.join(plot_dir, "step-{}-align.png".format(step)),
                                    title="{}, step={}, loss={:.5f}".format("Tacotron",
                                                                                
                                                                                step, lip2wav_loss.item()),
                                    max_len=target_length // hparams.outputs_per_step)
                # save real and predicted mel-spectrogram plot to disk (control purposes)
                plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir,
                                                                    "step-{}-mel-spectrogram.png".format(
                                                                        step)),
                                        title="{}, step={}, loss={:.5f}".format("Tacotron",
                                                                                
                                                                                    step, lip2wav_loss.item()),
                                        target_spectrogram=target,
                                        max_len=target_length)
                    # log("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
                    
                print("Saving loss data")

                print("Prepare Training Losses")
                lip2wav_loss_data = {
                    'Total_loss' : lip2wav_loss_array,
                    'before_postnet_loss' : before_postnet_loss_array,
                    'after_postnet_loss' : after_postnet_loss_array,
                    'l1_loss' : lip2wav_l1_loss_array,
                    'regularization_array' : regularization_array
                }
                wav2lip_loss_data = {
                    'Total_loss' : wav2lip_loss_array,
                    'sync_loss' : sync_loss_array,
                    'percep_loss' : percep_loss_array,
                    'l1_loss' : wav2lip_l1_loss_array,
                    'real_loss' : real_loss_array, 
                    'fake_loss' : fake_loss_array
                }
                cycle_loss_data = {
                    'cycle_loss_A' : cycle_loss_A_array,
                    'cycle_loss_B' : cycle_loss_B_array
                }                
                save_loss_data('lip2wav',os.path.join( args.checkpoint_dir_loss_data, 'lip2wav', 'train'), lip2wav_loss_data, global_step)
                save_loss_data('wav2lip',os.path.join( args.checkpoint_dir_loss_data, 'wav2lip', 'train'), wav2lip_loss_data, global_step)
                save_loss_data('cycle'  ,os.path.join( args.checkpoint_dir_loss_data, 'cycle'), cycle_loss_data, global_step)

                print("Prepare Validation Losses")
                lip2wav_validation_data = {
                    'Total_loss' : avg_val_lip2wav_loss_array,
                    'before_postnet_loss' : avg_val_before_postnet_loss_array,
                    'after_postnet_loss' : avg_val_after_postnet_loss_array,
                    'l1_loss' : avg_val_lip2wav_l1_loss_array,
                    'regularization_array' : avg_val_regularization_array
                }
                wav2lip_validation_data = {
                    'Total_loss' : avg_val_wav2lip_loss_array,
                    'sync_loss' : avg_val_sync_loss_array,
                    'percep_loss' : avg_val_percep_loss_array,
                    'l1_loss' : avg_val_wav2lip_l1_loss_array,
                    'real_loss' : avg_val_real_loss_array, 
                    'fake_loss' : avg_val_fake_loss_array
                }

                save_loss_data('lip2wav_val', os.path.join(args.checkpoint_dir_loss_data, 'lip2wav', 'val'), lip2wav_validation_data, global_step)
                save_loss_data('wav2lip_Val', os.path.join(args.checkpoint_dir_loss_data, 'wav2lip', 'val'), wav2lip_validation_data, global_step)
            
            if hparams.tacotron_decay_learning_rate and (global_step >= hparams.tacotron_start_decay) and cyclemodel.lip2wav_optim.param_groups[0]['lr'] >= hparams.tacotron_final_learning_rate:
                scheduler_lip2wav.step()


            # if global_step % hparams.eval_interval == 0:
            #     with torch.no_grad():
            #         average_sync_loss = eval_model(test_data_loader, global_step, device, cyclemodel, disc)

            #         if average_sync_loss < .75:
            #             hparams.set_hparam('syncnet_wt', 0.03)

            print('\n Lip2wav: Total loss {} \n Cycle Loss: A {}, B {} \n Wav2lip: Total loss {} \n Discriminator: Fake loss: {}, Real loss: {} \n Grad Norm: lip2wav {}, wav2lip {} \n'.format(running_lip2wav_loss / (step + 1),
                                                                                                                                            running_cycle_loss_A / (step+1),
                                                                                                                                            running_cycle_loss_B / (step+1),
                                                                                                                                            running_wav2lip_loss / (step + 1),
                                                                                                                                            running_disc_fake_loss / (step + 1),
                                                                                                                                            running_disc_real_loss / (step + 1),
                                                                                                                                            grad_norm_lip2wav,
                                                                                                                                            grad_norm_wav2lip))
        
        


        global_epoch += 1

def save_loss_data(model_name, ckpt_dir, data, global_step):
    full_name = os.path.join(ckpt_dir, 'loss_data_{}_{:09d}.npz'.format(model_name, global_step))
    np.savez(full_name, **data)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def save_checkpoint_lip2wav(model, optimizer, iteration, n_gpu):
    ckpt_pth = os.path.join(args.checkpoint_dir_lip2wav, 'ckpt_{}'.format(iteration))
    torch.save({'model': (model.module if n_gpu > 1 else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration}, ckpt_pth)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint_wav2lip(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

def load_checkpoint_lip2wav(ckpt_pth, model, optimizer, device, n_gpu):
    ckpt_dict = torch.load(ckpt_pth, map_location = device)
    (model.module if n_gpu > 1 else model).load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    iteration = ckpt_dict['iteration']
    return model, optimizer, iteration

if __name__ == "__main__":
    checkpoint_dir_wav2lip = args.checkpoint_dir_wav2lip
    checkpoint_dir_lip2wav = args.checkpoint_dir_lip2wav

    rank = local_rank = 0
    n_gpu = 1
    if 'WORLD_SIZE' in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(hparams.n_workers)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        n_gpu = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend = 'nccl', rank = local_rank, world_size = n_gpu)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))


    use_cuda = torch.cuda.is_available()
    print('use_cuda: {}'.format(use_cuda))

    logloss = nn.BCELoss()

    syncnet = SyncNet().to(device)
    for p in syncnet.parameters():
        p.requires_grad = False

    recon_loss = nn.L1Loss()

    # Dataset and Dataloader setup
    train_data_loader = prepare_dataloaders('train')
    test_data_loader = prepare_dataloaders('val')

    device = torch.device("cuda" if use_cuda else "cpu")

    #wav2lip Model
    wav2lip = Wav2Lip().to(device)
    disc = Wav2Lip_disc_qual().to(device)
    if n_gpu > 1:
        wav2lip = torch.nn.parallel.DistributedDataParallel(
            wav2lip, device_ids = [local_rank])

    if n_gpu > 1:
        disc = torch.nn.parallel.DistributedDataParallel(
            disc, device_ids = [local_rank])

    #lip2wav model
    lip2wav = Lip2Wav().to(device)
    if n_gpu > 1:
        lip2wav = torch.nn.parallel.DistributedDataParallel(
            lip2wav, device_ids = [local_rank])


    print('Total wav2lip trainable params {}'.format(sum(p.numel() for p in wav2lip.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))
    print('Total Lip2Wav trainable params {}'.format(sum(p.numel() for p in lip2wav.parameters() if p.requires_grad)))

    ## Wav2lip Optimizers
    wav2lip_optimizer = optim.Adam([p for p in wav2lip.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))
    
    ## Lip2wav Optimizers
    lip2wav_optimizer = optim.Adam(lip2wav.parameters(), lr = hparams.tacotron_initial_learning_rate,
                                betas = (hparams.tacotron_adam_beta1, hparams.tacotron_adam_beta2), eps = hparams.tacotron_adam_epsilon,
                                )

    if args.checkpoint_path_wav2lip is not None:
        wav2lip = load_checkpoint_wav2lip(args.checkpoint_path_wav2lip, wav2lip, wav2lip_optimizer, reset_optimizer=False)

    if args.disc_checkpoint_path is not None:
        disc = load_checkpoint_wav2lip(args.disc_checkpoint_path, disc, disc_optimizer, 
                                reset_optimizer=False, overwrite_global_states=False)
    
    if args.checkpoint_path_lip2wav is not None:
        lip2wav, lip2wav_optimizer, _ = load_checkpoint_lip2wav(args.checkpoint_path_lip2wav, lip2wav, lip2wav_optimizer, device, n_gpu)

    if not os.path.exists(checkpoint_dir_wav2lip):
        os.mkdir(checkpoint_dir_wav2lip)

    ## Define loss calcs
    sync_loss = Wav2LipLoss(syncnet, device, hparams.T)
    tacotron_loss = Tacotron2Loss()
    cyclemodel = cyclelip(device, n_gpu, lip2wav, wav2lip, disc, lip2wav_optimizer, wav2lip_optimizer, disc_optimizer, tacotron_loss, sync_loss)

    # Train!
    train(device, cyclemodel, train_data_loader, test_data_loader,
              checkpoint_dir_wav2lip=checkpoint_dir_wav2lip,
              checkpoint_dir_lip2wav=checkpoint_dir_lip2wav,
              checkpoint_interval=args.checkpoint_interval,
              nepochs=hparams.nepochs, data_log_interval=args.data_log_interval, hparams = hparams)
    







































































































    # The souls who have worked on this project, enter your name if you are working on this!
    # 1 - Jared >:3 - You've made it this far all the best in continuing to improve this! - jaredgoh02@gmail.com
