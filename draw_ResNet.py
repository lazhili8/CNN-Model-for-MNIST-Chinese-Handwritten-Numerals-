with open("log/ResNet-0.001-100", "r") as f:
    data = f.readlines()

epoch = []
epoch_count = 0

for i in data[22::24]:
    epoch_count = epoch_count + 1
    epoch.append(epoch_count)

loss = []

for i in data[22::24]:
    j = (i[-9:-1])
    loss.append(float(j))

accuracy = []

for i in data[23::24]:
    j = (i[-7:-1])
    accuracy.append(float(j))

print(loss)
print(accuracy)

import matplotlib.pyplot as plt


plt.plot(epoch, accuracy, color='green', linewidth=5, label='test_accuracy')
plt.plot(epoch, loss, color='red', linewidth=5, label='train_loss')


#设置图表标题，并给坐标轴加上标签
plt.title('ResNet', fontsize=24)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('percentage', fontsize=14)
#plt.text(0,47,'loss:blue,accuracy:green')
# 设置刻度标记的大小
plt.tick_params(axis='both', labelsize=14)
plt.legend()
plt.show()
