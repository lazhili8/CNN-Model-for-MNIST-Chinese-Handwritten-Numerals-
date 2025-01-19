import random

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

source_data = images
source_label = targets
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
data = Data_Loader.GetLoader(source_data, source_label)

order = [i for i in range(15000)]
random.shuffle(order)

train_set = []
test_set = []

for i in range(13500):
    train_set.append(data[order[i]])
for i in range(13500, 15000):
    test_set.append(data[order[i]])

# 一次性加载BATCH_SIZE个打乱顺序的数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 构建卷积神经网络模型


cnn = CNN_model.CNN().to(DEVICE)

optimizer = torch.optim.SGD(cnn.parameters(), lr=0.00001, momentum=0.9)

train_loss = []
test_loss = []
_accuracy = []

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()  # 模型训练
    global train_loss
    for batch_index, (data, target) in enumerate(train_loader):  # 一批中的一个，（图片，标签）
        data, target = data.to(device), target.to(device)  # 部署到DEVICE上去
        optimizer.zero_grad()  # 梯度初始化为0
        output = model(data)  # 训练后的结果
        loss = function.cross_entropy(output, target)  # 多分类计算损失函数
        loss.backward()  # 反向传播 得到参数的梯度参数值
        optimizer.step()  # 参数优化
        if batch_index % 3000 == 0:  # 每3000个打印一次
            print("Train Epoch: {} \t Loss:{:.6f}".format(epoch, loss.item()))
            train_loss.append(loss.item())


def test_model(model, device, text_loader):
    model.eval()  # 模型验证
    correct = 0.0  # 正确
    accuracy = 0.0  # 正确率
    text_loss = 0.0
    global test_loss
    global _accuracy
    with torch.no_grad():  # 不会计算梯度，也不会进行反向传播
        for data, target in text_loader:
            data, target = data.to(device), target.to(device)  # 部署到device上
            output = model(data)  # 处理后的结果
            text_loss += function.cross_entropy(output, target).item()  # 计算测试损失之和
            pred = output.argmax(dim=1)  # 找到概率最大的下标（索引）
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累计正确的次数
        text_loss /= len(test_loader.dataset)  # 损失和/数据集的总数量 = 平均loss
        accuracy = 100.0 * correct / len(text_loader.dataset)  # 正确个数/数据集的总数量 = 正确率
        print("Test__Average loss: {:4f},accuracy: {:.3f}\n".format(text_loss, accuracy))
        test_loss.append(text_loss)
        _accuracy.append(accuracy)


for epoch in range(1, EPOCHS + 1):
    train_model(cnn, DEVICE, train_loader, optimizer, epoch)
    test_model(cnn, DEVICE, test_loader)

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
_accuracy = np.array(_accuracy)

torch.save(cnn.state_dict(), 'weight/CNN_model.ckpt')  # 保存为model.ckpt
