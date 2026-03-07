import torch

def _detect_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = _detect_best_device()
use_gpu = device.type != 'cpu'

def set_use_gpu(v):
    global use_gpu
    global device
    use_gpu = v
    if not use_gpu:
        device = torch.device('cpu')

def get_use_gpu():
    global use_gpu
    return use_gpu

def set_device(d):
    global device
    global use_gpu
    device = d
    use_gpu = device.type != 'cpu'

def get_device():
    global device
    return device
