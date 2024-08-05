"""
@FileName：train.py
@Description：
@Author：wbzuo
@Time：2024-06-26 17:43
"""
import time

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_dataloader
from model.basemodels import UNet

import os

from utils.config import parse_args
from utils.metric import SegmentationMetric
from utils.seed import seed_everything

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

weight_path = './checkpoint/unet_weights.pth'

# 训练一个阶段
def train_one_epoch(model, device, criterion, optimizer, train_loader):
    model.train()
    epoch_loss = 0  # 计算每一轮次的
    num_total = 0
    pbar = tqdm(train_loader, total=len(train_loader))
    for images, labels in pbar:

        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        epoch_loss += loss.item()
        num_total += images.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 设置进度条
        pbar.set_postfix(loss=epoch_loss/num_total)

    lr = optimizer.param_groups[0]['lr']
    return epoch_loss / num_total, lr

def val_one_epoch(model, device, num_classes,criterion, val_loader):
    model.eval()
    epoch_loss = 0
    num_total = 0
    pbar = tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            pred = F.softmax(logits)
            pred= torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()

            # 评价指标
            metric = SegmentationMetric(num_classes=num_classes)
            metric.addBatch(pred, labels.cpu().numpy())
            pa = metric.pixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()

            epoch_loss += loss.item()
            num_total += images.size(0)

            # 设置进度条
            pbar.set_postfix(loss=epoch_loss / num_total)


    return epoch_loss / num_total, pa, mIoU

def train(model, device, criterion, optimizer, train_loader, val_loader, epochs = 20):

    # 开始训练
    best_mean_iou = 0.0
    train_loss_list = []
    val_loss_list = []
    mean_iou_list = []
    pa_list = []
    since = time.time()
    for epoch in range(epochs):
        # 训练
        train_loss, _ = train_one_epoch(model, device, criterion, optimizer, train_loader)
        train_loss_list.append(train_loss)

        # 评估
        val_loss, pa, mIoU = val_one_epoch(model, device, 21,criterion, val_loader)
        val_loss_list.append(val_loss)
        pa_list.append(pa)
        mean_iou_list.append(mIoU)




        end = time.time()
        print(f"[{epoch + 1}/{epochs} 耗时：{(end - since):.4f}s]\n" +
              f"  mean train loss: [{train_loss:.4f}],\n" +
              f"  mean val loss: [{val_loss:.4f}],\n" +
              f"  pa: [{pa:.4f}],\n" +
              f"  mIoU: [{mIoU:.4f}]")

    # 结束训练
    end = time.time()
    print(f"{epochs} 轮训练共耗时： {(end - since):}")





if __name__ == '__main__':
    # ===== 命令行参数 =====
    args = parse_args()

    # ===== 随机种子 =====
    seed_everything(22)


    # =====数据部分======
    batch_size = 4
    data_path = r'D:/232/ai/resources/datasets/VOC2012'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = get_dataloader("train", batch_size=batch_size)
    val_loader = get_dataloader("val", batch_size=batch_size)
    n_classes = 21

    # ===== 设备部分 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 模型部分 =====
    model = UNet(n_channels=3, n_classes=n_classes).to(device)
    # if os.path.exists(weight_path):
    #     UNet.load_state_dict(torch.load(weight_path))
    #     print('successfully loaded!')
    # else:
    #     print('no weights')

    # ====== 训练轮次 =====
    epochs = 100

    # ====== 优化器 =====
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    # ====== 损失函数 =====
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 训练
    train(model = model,
          device=device,
          criterion = criterion,
          optimizer = optimizer,
          train_loader = train_loader,
          val_loader = val_loader,
          epochs = epochs)






