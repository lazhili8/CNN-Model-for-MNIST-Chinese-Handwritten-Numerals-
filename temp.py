import random
from ctypes.wintypes import RGB

from torchvision.transforms import transforms

import CNN_model
import Data_Loader

import pickle
import numpy as np

import torch
import torch.nn.functional as function

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

BATCH_SIZE = 64  # 每批处理的数据 一次性多少个
DEVICE = torch.device("cuda", 0)  # 使用CPU
EPOCHS = 1000  # 训练数据集的轮次

with open("chn_mnist", "rb") as f:
    data = pickle.load(f)
images = torch.Tensor(data["images"])
images = torch.unsqueeze(images, 1)

targets = data["targets"]
targets = targets.tolist()

for i in range(len(targets)):
    if targets[i] == 100:
        targets[i] = 11
    elif targets[i] == 1000:
        targets[i] = 12
    elif targets[i] == 10000:
        targets[i] = 13
    elif targets[i] == 100000000:
        targets[i] = 14

print(images[5])

new_img_PIL = transforms.ToPILImage()(images[5])
new_img_PIL.show() # 处理后的PIL图片