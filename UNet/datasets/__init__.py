"""
@FileName：__init__.py
@Description：
@Author：wbzuo
@Time：2024-08-02 16:15
"""
from torch.utils.data import DataLoader

from datasets.dataset import PASCAL_BSD


# 获取数据集
def get_dataloader(mode="train", batch_size=16, shuffle=True, num_workers=4):
    data = PASCAL_BSD(mode)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
