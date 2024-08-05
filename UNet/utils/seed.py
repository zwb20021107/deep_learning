"""
@FileName：seed.py
@Description：
@Author：wbzuo
@Time：2024-08-03 12:48
"""
import random

import numpy as np
import torch

#   设置种子
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#   设置Dataloader的种子
def worker_init_fn(worker_id, rank, seed = 11):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
