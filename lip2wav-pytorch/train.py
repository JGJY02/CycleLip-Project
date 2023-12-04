import os
# from re import split
import time
import torch
import argparse
import numpy as np
# from inference import infer
from utils.util import mode
from hparams import hparams as hps
from utils.logger import Tacotron2Logger
from utils.dataset import ljdataset, ljcollate
from model.model import Tacotron2, Tacotron2Loss
from torch.utils.data import DistributedSampler, DataLoader
from utils import plot
from utils import audio
np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)

#Jared Imports
from tqdm import tqdm
#
## Chat Gpt Code
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
        elif self.init_lr > hps.tacotron_final_learning_rate:
            decay_steps = self.decay_steps
            if self.last_epoch < self.tacotron_start_decay + self.warmup_steps:
                decay_steps -= (self.tacotron_start_decay + self.warmup_steps - self.last_epoch)
            lr = self.init_lr * (self.decay_rate ** (self.last_epoch // decay_steps))
        else:
            lr = hps.tacotron_final_learning_rate

        return [lr]

def prepare_dataloaders(split):
    print("dataset preperation has begun")
    trainset = ljdataset(args.data_root, split)
    collate_fn = ljcollate(hps.outputs_per_step)
    # sampler = DistributedSampler(trainset) if n_gpu > 1 else None
    if split == 'train':
        train_loader = DataLoader(trainset, num_workers = 4, shuffle = True,
                                batch_size = hps.tacotron_batch_size, collate_fn = collate_fn)
    else:
        train_loader = DataLoader(trainset, num_workers = 4, shuffle = True,
                                 batch_size = hps.tacotron_batch_size, collate_fn = collate_fn)
    
    return train_loader


def load_checkpoint(ckpt_pth, model, optimizer, device, n_gpu):
    ckpt_dict = torch.load(ckpt_pth, map_location = device)
    (model.module if n_gpu > 1 else model).load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    iteration = ckpt_dict['iteration']
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth, n_gpu):
    torch.save({'model': (model.module if n_gpu > 1 else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration}, ckpt_pth)

def eval_model(test_data_loader, criterion, n_gpu, model):
    
    eval_steps = 300
    cur_step = 0
    print('Evaluating across for {} steps'.format(eval_steps))
    total_loss, before_loss, after_loss, l1_loss, regularization_loss = [], [], [], [], []

    for batch in test_data_loader:
        model.eval()

        x, y = (model.module if n_gpu > 1 else model).parse_batch(batch)
        y_pred, trainable_params = model(x)

        # loss
        loss, items = criterion(y_pred, y, trainable_params)

        total_loss.append(loss.item())
        before_loss.append(items[0])
        after_loss.append(items[1])
        l1_loss.append(items[2])
        regularization_loss.append(items[3])

        cur_step += 1
        if cur_step > eval_steps: break
        
        


    print('Evaluation Results | Total loss: {} | before: {}, after: {}, l1: {} Regularization: {}'.format(sum(total_loss) / len(total_loss), sum(before_loss) / len(before_loss),
                                                        sum(after_loss) / len(after_loss),
                                                        sum(l1_loss) / len(l1_loss),
                                                        sum(regularization_loss) / len(regularization_loss)))
    
    avg_total_loss = sum(total_loss) / len(total_loss)
    avg_before_loss = sum(before_loss) / len(before_loss)
    avg_after_loss = sum(after_loss) / len(after_loss)
    avg_l1_loss = sum(l1_loss) / len(l1_loss)
    avg_regularization_loss = sum(regularization_loss) / len(regularization_loss)
    
    return [avg_total_loss, avg_before_loss, avg_after_loss, avg_l1_loss, avg_regularization_loss]

    
def train(args):
    # setup env
    rank = local_rank = 0
    n_gpu = 1
    plot_dir = os.path.join(args.checkpoint_dir_loss_data, "plots")
    wav_dir = os.path.join(args.checkpoint_dir_loss_data, "wavs")
    mel_dir = os.path.join(args.checkpoint_dir_loss_data, "mel-spectrograms")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)

    if 'WORLD_SIZE' in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(hps.n_workers)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        n_gpu = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend = 'nccl', rank = local_rank, world_size = n_gpu)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))

    # build model
    model = Tacotron2()
    mode(model, True)
    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids = [local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr = hps.tacotron_initial_learning_rate,
                                betas = (hps.tacotron_adam_beta1, hps.tacotron_adam_beta2), eps = hps.tacotron_adam_epsilon,
                                )

    criterion = Tacotron2Loss()
    
    # load checkpoint
    iteration = 1
    if args.checkpoint_path != '':
        model, optimizer, iteration = load_checkpoint(args.checkpoint_path, model, optimizer, device, n_gpu)
        iteration += 1

    # get scheduler
    if hps.tacotron_decay_learning_rate:
        # lr_lambda = lambda step: hps.tacotron_decay_steps**0.5*min((step+1)*hps.tacotron_decay_steps**-1.5, (step+1)**-0.5)
        # if args.checkpoint_path != '':
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
        # else:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler = ExponentialLRWithWarmup(optimizer, hps.tacotron_initial_learning_rate, hps.tacotron_start_decay,
                                    hps.tacotron_decay_steps, hps.tacotron_decay_rate, 0)

    # make dataset
    train_data_loader = prepare_dataloaders('train')
    test_data_loader = prepare_dataloaders('test')
    
    if rank == 0:
        # get logger ready
        if args.log_dir != '':
            if not os.path.isdir(args.log_dir):
                os.makedirs(args.log_dir)
                os.chmod(args.log_dir, 0o775)
            logger = Tacotron2Logger(args.log_dir)

        # get ckpt_dir ready
        if args.checkpoint_dir != '' and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
            os.chmod(args.checkpoint_dir, 0o775)

    model.train()

    # setup loss saving
    lip2wav_loss_array, before_postnet_loss_array, after_postnet_loss_array, lip2wav_l1_loss_array, regularization_array = [], [], [], [], []
    avg_val_lip2wav_loss_array, avg_val_before_postnet_loss_array, avg_val_after_postnet_loss_array, avg_val_lip2wav_l1_loss_array, avg_val_regularization_array = [], [], [], [], []


    # ================ MAIN TRAINNIG LOOP! ===================
    epoch = 0
    
    while iteration <= args.tacotron_train_steps:
        for batch in train_data_loader:
            if iteration > args.tacotron_train_steps:
                break
            start = time.perf_counter()

            model.train()
            optimizer.zero_grad()

            #Move Data to cuda Device
            x, y = (model.module if n_gpu > 1 else model).parse_batch(batch)
            y_pred, trainable_params = model(x)

            # loss
            loss, items = criterion(y_pred, y, trainable_params)

            # zero grad
            model.zero_grad()

            # backward, grad_norm, and update
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.tacotron_clip_gradients_clip_thresh)
            optimizer.step()
            
            if hps.tacotron_decay_learning_rate and (iteration >= hps.tacotron_start_decay) and optimizer.param_groups[0]['lr'] >= hps.tacotron_final_learning_rate:
                scheduler.step()
                #print(optimizer.param_groups[0]['lr'])
                

            dur = time.perf_counter()-start

            ## append losses for figure visualization later
            lip2wav_loss_array.append(loss.item())
            before_postnet_loss_array.append(items[0])
            after_postnet_loss_array.append(items[1])
            lip2wav_l1_loss_array.append(items[2])
            regularization_array.append(items[3])

            if rank == 0:
                # info
                
                print('Iter: {} Total Loss {:.2e} before Loss: {:.2e} after Loss: {:.2e} l1 loss: {:.2e} Regularization Loss {:.2e} Grad Norm: {:.2e} {:.1f}s/it'.format(
                    iteration, loss, items[0], items[1], items[2], items[3], grad_norm, dur))
                    

                    

                # log
                if args.log_dir != '' and (iteration % args.checkpoint_interval == 0 or iteration == args.tacotron_train_steps or \
                        iteration == 300):
                    learning_rate = optimizer.param_groups[0]['lr']
                    logger.log_training(items, grad_norm, learning_rate, iteration)

                if iteration % args.eval_interval == 0:
                    with torch.no_grad():
                        lip2wav_val_losses = eval_model(test_data_loader, criterion, n_gpu, model)
                        avg_val_lip2wav_loss_array.append(lip2wav_val_losses[0])
                        avg_val_before_postnet_loss_array.append(lip2wav_val_losses[1])
                        avg_val_after_postnet_loss_array.append(lip2wav_val_losses[2])
                        avg_val_lip2wav_l1_loss_array.append(lip2wav_val_losses[3])
                        avg_val_regularization_array.append(lip2wav_val_losses[4])



                # save ckpt
                if args.checkpoint_dir != '' and (iteration % args.checkpoint_interval == 0):
                    ckpt_pth = os.path.join(args.checkpoint_dir, 'ckpt_{}'.format(iteration))
                    save_checkpoint(model, optimizer, iteration, ckpt_pth, n_gpu)

                    # save predicted mel spectrogram to disk (debug)
                    step = iteration
                    mel_prediction = y_pred[1][0].cpu().detach().numpy()
                    hparams = hps
                    alignment = y_pred[2][0].cpu().detach().numpy()
                    target_length = y[2][0].cpu().detach().numpy()
                    target = y[0][0].cpu().detach().numpy()

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
                                                                                    
                                                                                    step, loss),
                                        max_len=target_length // hparams.outputs_per_step)
                    # save real and predicted mel-spectrogram plot to disk (control purposes)
                    plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir,
                                                                       "step-{}-mel-spectrogram.png".format(
                                                                           step)),
                                          title="{}, step={}, loss={:.5f}".format("Tacotron",
                                                                                    
                                                                                      step, loss),
                                          target_spectrogram=target,
                                          max_len=target_length)
                    # log("Input at step {}: {}".format(step, sequence_to_text(input_seq)))

                if iteration == 1 or iteration % args.data_log_interval == 0:
                    print("Saving loss data")

                    print("Prepare Training Losses")
                    lip2wav_loss_data = {
                        'Total_loss' : lip2wav_loss_array,
                        'before_postnet_loss' : before_postnet_loss_array,
                        'after_postnet_loss' : after_postnet_loss_array,
                        'l1_loss' : lip2wav_l1_loss_array,
                        'regularization_array' : regularization_array
                    }

                    save_loss_data('lip2wav',os.path.join( args.checkpoint_dir_loss_data, 'train'), lip2wav_loss_data, iteration)
   

                    print("Prepare Validation Losses")
                    lip2wav_validation_data = {
                        'Total_loss' : avg_val_lip2wav_loss_array,
                        'before_postnet_loss' : avg_val_before_postnet_loss_array,
                        'after_postnet_loss' : avg_val_after_postnet_loss_array,
                        'l1_loss' : avg_val_lip2wav_l1_loss_array,
                        'regularization_array' : avg_val_regularization_array
                    }


                    save_loss_data('lip2wav_val', os.path.join(args.checkpoint_dir_loss_data, 'val'), lip2wav_validation_data, iteration)

            iteration += 1
        epoch += 1

    if rank == 0 and args.log_dir != '':
        logger.close()


def save_loss_data(model_name, ckpt_dir, data, global_step):
    full_name = os.path.join(ckpt_dir, 'loss_data_{}_{:09d}.npz'.format(model_name, global_step))
    np.savez(full_name, **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to train the lip2wav model on the pytorch Architecture')
    # path
    parser.add_argument('-l', '--log_dir', type = str, default = 'log',
                        help = 'directory to save tensorboard logs')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default='', type=str)
    parser.add_argument("--summary_interval", type=int, default=2500,
                        help="Steps between running summary ops")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, # Was 5000
                        help="Steps between writing checkpoints")
    parser.add_argument("--eval_interval", type=int, default=1000, # Was 10000
                        help="Steps between eval on test data")
    parser.add_argument("--tacotron_train_steps", type=int, default=1000000, # Was 100000
                        help="total number of tacotron training steps")
    parser.add_argument('--checkpoint_dir_loss_data', help='this is where the loss data is stored', default = '/fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Lip2Wav-pytorchConversion/LIP2WAV-Converted/training_results/loss_data')
    parser.add_argument("--data_log_interval", type=int, default=5000, # Was 5000
                    help="Steps between data logs")

    args = parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(args)
