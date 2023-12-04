import torch
import numpy as np
import sys, cv2, os, pickle, argparse
from tqdm import tqdm
from shutil import copy
from glob import glob
from model.model import Tacotron2
from hparams import hparams as hps
from utils import audio, dataset 
from utils.util import mode, to_arr


class Generator(object):
	def __init__(self, model):
		super(Generator, self).__init__()

		self.model = model
		


	def read_window(self, window_fnames):
		window = []
		for fname in window_fnames:
			img = cv2.imread(fname)
			if img is None:
				raise FileNotFoundError('Frames maybe missing in {}.' 
						' Delete the video to stop this exception!'.format(sample['folder']))

			img = cv2.resize(img, (hps.img_size, hps.img_size))
			window.append(img)

		images = np.asarray(window) / 255. # T x H x W x 3
		images = images.transpose(3, 0, 1, 2) # For pytorch implementation channel must be last
		return images

	def vc(self, sample, outfile):
		id_windows = [range(i, i + hps.T) for i in range(0, (sample['till'] // hps.T) * hps.T, 
					hps.T - hps.overlap) if (i + hps.T <= (sample['till'] // hps.T) * hps.T)]

		all_windows = [[sample['folder'].format(id) for id in window] for window in id_windows]
		last_segment = [sample['folder'].format(id) for id in range(sample['till'])][-hps.T:]
		all_windows.append(last_segment)

		ref = np.load(os.path.join(os.path.dirname(sample['folder']), 'ref.npz'))['ref'][0]
		ref = torch.FloatTensor(ref)
        

		for window_idx, window_fnames in enumerate(all_windows):
			images = self.read_window(window_fnames)
			images = np.expand_dims(images, axis=0)
			images = torch.FloatTensor(images)
			
			s = self.model.inference(images, ref)[0]
			s = torch.cat(s)
			s = s.cpu().detach().numpy()
			if window_idx == 0:
				mel = s
				
			elif window_idx == len(all_windows) - 1:
				remaining = ((sample['till'] - id_windows[-1][-1] + 1) // 5) * 16
				if remaining == 0:
					continue
				mel = np.concatenate((mel, s[:, -remaining:]), axis=1)
			else:
				mel = np.concatenate((mel, s[:, hps.mel_overlap:]), axis=1)

		wav = audio.inv_mel_spectrogram(mel)
		

		audio.save_wav(wav, outfile, sr=hps.sample_rate)
		
def get_image_list(split, data_root):
    assert split in ['test', 'val', 'train', '*']
    filelist = list(glob(os.path.join(data_root, '{}/*/*/*/*.jpg'.format(split))))

    print('Test filelist contains {} images'.format(len(filelist)))
    
    return filelist

def get_vidlist(data_root):
	test = get_image_list('test', data_root)
	test_vids = {}
	for x in test:
		x = x[:x.rfind('/')]
		if len(os.listdir(x)) < 30: continue
		test_vids[x] = True
	return list(test_vids.keys())

def complete(folder):
	# first check if ref file present
	if not os.path.exists(os.path.join(folder, 'ref.npz')):
		return False

	frames = glob(os.path.join(folder, '*.jpg'))
	ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames]
	sortedids = sorted(ids)
	if sortedids[0] != 0: return False
	for i, s in enumerate(sortedids):
		if i != s:
			return False
	return True

def load_model(ckpt_pth):
    ckpt_dict = torch.load(ckpt_pth)
    model = Tacotron2()
    model.load_state_dict(ckpt_dict['model'])
    model = mode(model, True).eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_root", help="Speaker folder path", required=True)
    parser.add_argument('-r', "--results_root", help="Speaker folder path", required=True)
    parser.add_argument('--checkpoint', help="Path to trained checkpoint", required=True)
    args = parser.parse_args()

    #Setup Model
    rank = local_rank = 0
    n_gpu = 1
    if 'WORLD_SIZE' in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(hps.n_workers)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        n_gpu = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend = 'nccl', rank = local_rank, world_size = n_gpu)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))
    model = Tacotron2()

    lip2wav_model = load_model(args.checkpoint)
    videos = get_vidlist(args.data_root)

    RESULTS_ROOT = args.results_root
    if not os.path.isdir(RESULTS_ROOT):
        os.mkdir(RESULTS_ROOT)

    GTS_ROOT = os.path.join(RESULTS_ROOT, 'gts/')
    WAVS_ROOT = os.path.join(RESULTS_ROOT, 'wavs/')
    files_to_delete = []
    if not os.path.isdir(GTS_ROOT):
        os.mkdir(GTS_ROOT)
    else:
        files_to_delete = list(glob(GTS_ROOT + '*'))
    if not os.path.isdir(WAVS_ROOT):
        os.mkdir(WAVS_ROOT)
    else:
        files_to_delete.extend(list(glob(WAVS_ROOT + '*')))
    for f in files_to_delete: os.remove(f)

    g = Generator(lip2wav_model)
    for vid in tqdm(videos):
        if not complete(vid):
            continue

        sample = {}
        vidpath = vid + '/'

        sample['folder'] = vidpath + '{}.jpg'

        images = glob(vidpath + '*.jpg')
        sample['till'] = (len(images) // 5) * 5

        vidname = vid.split('/')[-2] + '_' + vid.split('/')[-1]
        outfile = WAVS_ROOT + vidname + '.wav'
        g.vc(sample, outfile)

        copy(vidpath + 'audio.wav', GTS_ROOT + vidname + '.wav')