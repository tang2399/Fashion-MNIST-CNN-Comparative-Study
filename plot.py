# 数据集的下载与显示
from torchvision.datasets import FashionMNIST  # 加载数据库
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
train_data = FashionMNIST(root='./dataset',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

# 获取每一个Batch的数据
b_x = b_y = 0
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第一维后转换为Numpy数组方便画图
batch_y = b_y.numpy()  # 将张量转换为Numpy数组
class_label = train_data.classes  # 读取数据集标签
print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(14, 6))
plt.suptitle("Sample Graph", fontsize=27)
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.1)

plt.tight_layout()
plt.savefig('./images/SampleGraph.png')
plt.show()



