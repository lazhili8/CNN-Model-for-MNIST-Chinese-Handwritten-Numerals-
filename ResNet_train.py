import random
import Data_Loader

import pickle

import torch
import torch.nn.functional as function
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader

import argparse

import ResNet_model

BATCH_SIZE = 64  # 每批处理的数据 一次性多少个
DEVICE = torch.device("cuda", 1)  # 使用CUDA
EPOCHS = 100  # 训练数据集的轮次

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


# 6.定义优化器
model = ResNet_model.ResNet18().to(DEVICE)  # 创建模型并将模型加载到指定设备上

optimizer = optim.Adam(model.parameters(), lr=0.00001)  # 优化函数

criterion = nn.CrossEntropyLoss()


# 7.训练
def train_model(model, device, train_loader, optimizer, epoch):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    model.train()  # 模型训练
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 部署到DEVICE上去
        optimizer.zero_grad()  # 梯度初始化为0
        output = model(data)  # 训练后的结果
        loss = criterion(output, target)  # 多分类计算损失
        loss.backward()  # 反向传播 得到参数的梯度值
        optimizer.step()  # 参数优化
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))
            if args.dry_run:
                break


# 8.测试
def test_model(model, device, text_loader):
    model.eval()  # 模型验证
    correct = 0.0  # 正确率
    global Accuracy
    text_loss = 0.0
    with torch.no_grad():  # 不会计算梯度，也不会进行反向传播
        for data, target in text_loader:
            data, target = data.to(device), target.to(device)  # 部署到device上
            output = model(data)  # 处理后的结果
            text_loss += criterion(output, target).item()  # 计算测试损失
            pred = output.argmax(dim=1)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累计正确的值
        text_loss /= len(test_loader.dataset)  # 损失和/加载的数据集的总数
        Accuracy = 100.0 * correct / len(text_loader.dataset)
        print("Test__Average loss: {:4f},Accuracy: {:.3f}\n".format(text_loss, Accuracy))


# 9.调用

for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)

torch.save(model.state_dict(), 'weight/ResNet_model.ckpt')
