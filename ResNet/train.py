"""
@FileName：train.py
@Description：
@Author：wbzuo
@Time：2024-05-22 0:49
"""
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from datasets.dataset import DogVSCatDataset
from models.basemodel import ResNet, BasicBlock, Bottleneck, resnet34, resnet18

data_transform = {
    "train": transforms.Compose([
        # transforms.RandomResizedCrop(224),
        #                          transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 在对图像进行标准化处理时，标准化参数来自于官网所提供的tansfer learning教程
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

    # Resize()函数，输入可能是sequence（元组类型，输入图像高和宽），也可能是int（将最小边缩放到指定的尺寸）
    "val": transforms.Compose([
        transforms.Resize((224,224)),  # 保持原图片长宽比不变，将最短边缩放到256
        # transforms.CenterCrop(224),  # 中心裁剪一个224×224的图片
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

transform = transforms.Compose([transforms.Resize((224, 224)),  # 把给定的图片resize到given size
                           transforms.ToTensor(),
                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 用均值和标准差归一化张量图像，把0-1变换到(-1,1).
                           ])



def train():
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 10


    train_dataset = DogVSCatDataset(mode="train",dir_path='D:/datasets/classification/猫狗分类/train', transform=data_transform['train'])
    val_dataset = DogVSCatDataset(mode="train", dir_path='D:/datasets/classification/猫狗分类/test', transform=data_transform['val'])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = resnet34(2).to(device)



    model_weight_path = "checkpoints/catVSdog/resnet34.pth"
    if os.path.isfile(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path))

    # define loss function
    criterion = nn.CrossEntropyLoss()


    optimizer  = optim.Adam(model.parameters(), lr = 0.1)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_acc = 0.0
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0


        for idx, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print train process
            rate = (idx + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss:{:^3.0f}%[{}—>{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.max(outputs.data, 1)[1]
                acc += torch.eq(preds, labels).sum().item()

                val_loader.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / len(train_loader), val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), model_weight_path)



def train_bee():
    batch_size = 64
    # num_workers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 50

    train_dataset = ImageFolder(r"D:\datasets\classification\hymenoptera_data\train", transform=transform)
    val_dataset = ImageFolder(r"D:\datasets\classification\hymenoptera_data\val", transform=transform)
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)
    #
    # for images, labels in train_loader:
    #     print(images.size(), labels.size())

    model = resnet34()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    model_weight_path = "checkpoints/bee/resnet34.pth"
    # if os.path.isfile(model_weight_path):
    #     model.load_state_dict(torch.load(model_weight_path))

    # define loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_acc = 0.0
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0

        for idx, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print train process
            rate = (idx + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss:{:^3.0f}%[{}—>{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.max(outputs.data, 1)[1]
                acc += torch.eq(preds, labels).sum().item()

                val_loader.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / len(train_loader), val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), model_weight_path)


if __name__ == '__main__':
    train_bee()




