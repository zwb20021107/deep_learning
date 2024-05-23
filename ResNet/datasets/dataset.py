"""
@FileName：dataset.py
@Description：
@Author：wbzuo
@Time：2024-05-22 0:50
"""
import os

import torch
import torch.utils.data as data
from PIL import Image





class DogVSCatDataset(data.Dataset):

    def __init__(self, transform=None, dir_path='D:/datasets/classification/dog&cat/train'):
        self.transform = transform
        self.list_img = []
        self.list_label = []
        self.data_size = 0

        for category in os.listdir(dir_path):
            category_path = os.path.join(dir_path, category)
            for img_path in os.listdir(category_path):
                img_path = os.path.join(category_path, img_path)

                img_label = 1 if category == "dog" else 0
                self.list_img.append(img_path)
                self.list_label.append(img_label)
                self.data_size += 1


    def __getitem__(self, idx):  # 重载data.Dataset父类方法，获取数据集中数据内容

        img = self.transform(Image.open(self.list_img[idx]))  # 打开图片
        label = self.list_label[idx]  # 获取image对应的label
        return img, label


    def __len__(self):
        return self.data_size               # 返回数据集批次大小



class DogVSCatDataset(data.Dataset):

    def __init__(self, mode='train', transform=None, dir_path='./datasets/dog&cat/train'):
        self.mode = mode
        self.transform = transform
        self.list_img = []
        self.list_label = []
        self.data_size = 0

        if mode == 'train':
            for  file in  os.listdir(dir_path):
                img_path = os.path.join(dir_path, file)
                self.list_img.append(img_path)
                name = file.split(sep='.')
                self.data_size += 1
                if name[0] == 'cat':
                    self.list_label.append(0)  # 图片为猫，label为0
                else:
                    self.list_label.append(1)  # 图片为狗，label为1，注意：list_img和list_label中的内容是一一配对的
        elif self.mode == 'test':  # 测试集模式下，只需要提取图片路径就行
            for file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, file)
                self.list_img.append(img_path)  # 添加图片路径至image list
                self.data_size += 1
                self.list_label.append(2)  # 添加2作为label，实际未用到，也无意义
        else:
            print('Undefined Dataset!')

    def __getitem__(self, item):  # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':  # 训练集模式下需要读取数据集的image和label
            img = self.transform(Image.open(self.list_img[item])) # 打开图片
            label = self.list_label[item]  # 获取image对应的label
            return img, label  # 将image和label转换成PyTorch形式并返回
        elif self.mode == 'test':  # 测试集只需读取image
            img = self.transform(Image.open(self.list_img[item])) # 打开图片
            return img  # 只返回image
        else:
            print('None')

    def __len__(self):
        return self.data_size               # 返回数据集大小

