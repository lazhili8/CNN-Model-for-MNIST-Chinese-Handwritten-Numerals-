import cv2 as cv
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

import CNN_model

path_raw = 'photo/raw/'
path_new = 'photo/new/'

print("请输入文件名称：")
file = input()

src = cv.imread(path_raw + file)  # 读取图片
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 白底黑字转换为黑底白字
cv.imwrite(path_new + file, binary)  # 将图像数据写入到图像文件中


def imag():  # 调整图片大小
    plt.ion()
    im = plt.imread(path_new + file)  # 读入图片
    images = Image.open(path_new + file)  # 将图片存储到images里面
    images = images.resize((64, 64))  # 调整图片的大小为28*28
    images = images.convert('L')  # 灰度化

    transform = transforms.ToTensor()  # 转换为tentor
    images = transform(images)  # 对图片进行transform
    images = images.reshape(1, 1, 64, 64)  # 调整图片尺寸（四维）

    # 加载网络和参数
    model = CNN_model.CNN()  # 加载模型
    model.load_state_dict(torch.load('weight/CNN_model.ckpt'))  # 加载参数
    model.eval()  # 测试模型
    outputs = model(images)  # 输出结果

    label = outputs.argmax(dim=1)  # 返回最大概率值的下标

    if int(label) == 0:
        str = '零'
    elif int(label) == 1:
        str = '一'
    elif int(label) == 2:
        str = '二'
    elif int(label) == 3:
        str = '三'
    elif int(label) == 4:
        str = '四'
    elif int(label) == 5:
        str = '五'
    elif int(label) == 6:
        str = '六'
    elif int(label) == 7:
        str = '七'
    elif int(label) == 8:
        str = '八'
    elif int(label) == 9:
        str = '九'
    elif int(label) == 10:
        str = '十'
    elif int(label) == 11:
        str = '百'
    elif int(label) == 12:
        str = '千'
    elif int(label) == 13:
        str = '万'
    elif int(label) == 14:
        str = '亿'
    print('预测结果：{}'.format(str))

imag()
