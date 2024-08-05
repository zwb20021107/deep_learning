"""
@FileName：config.py
@Description：
@Author：wbzuo
@Time：2024-08-03 13:11
"""
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int,help="number of total epochs to train")
    parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--lrf',type=float,default=0.1)
    args = parser.parse_args()

    return args