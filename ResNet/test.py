"""
@FileName：test.py
@Description：
@Author：wbzuo
@Time：2024-05-22 0:51
"""

import os
import json

import torch
from PIL import Image
from torchvision import transforms

from datasets.dataset import DogVSCatDataset
from models.basemodel import resnet34
from torch.utils.data import DataLoader


def test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹

    test_dataset = DogVSCatDataset(dir_path='D:/datasets/classification/猫狗分类/test',
                                  transform=transform)
    test_num = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=test_num)


    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights
    weights_path = "checkpoints/catVSdog/resnet34.pth"

    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    with torch.no_grad():
        acc = 0.0

        for idx, data in enumerate(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.max(outputs.data, 1)[1]
            print(labels)
            print(preds)

            acc += torch.eq(preds, labels).sum().item()

    test_accurate = acc / test_num
    print('accuracy: %.3f' %
          (test_accurate))

if __name__ == '__main__':
    test()