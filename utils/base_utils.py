import numpy as np
import os
import torch
from pathlib import Path
import random

import torchvision.transforms as transforms
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
my_norm_transforms = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])
def norm_transforms(image):
    return my_norm_transforms(image)



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)



def log_write(file, log):
    string = ''
    for key, value in log.items():
        string += key + '\t:'+str(value) +'\n'
    string += '\n\n\n'
    file.write(string)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
