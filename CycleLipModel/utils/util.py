import torch
import numpy as np
from hparams import hparams as hps

def mode(obj, model = False):
    if model and hps.is_cuda:
        obj = obj.cuda()
    elif hps.is_cuda:
        obj = obj.cuda(non_blocking = hps.pin_mem)
    return obj

def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask
