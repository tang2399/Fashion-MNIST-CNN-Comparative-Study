import copy
import time
import torch
from torchvision.datasets import FashionMNIST  # 导入数据库
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt

from model.LeNet import LeNet  # 导入模型
from model.AlexNet import AlexNet
from model.VGGNet16 import VGG16
from model.GoogLeNet import GoogLeNet
from model.ResNet18 import ResNet18

import torch.nn as nn
import pandas as pd


# 处理训练集和验证集，即数据加载的过程
def train_val_data_process():
    train_data = FashionMNIST(root='./dataset',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=model.size_input),
                                                            transforms.ToTensor()]),
                              download=True)

    # 划分数据集
    train_data, val_data = data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = data.DataLoader(dataset=train_data,
                                       batch_size=32,  # 一批次样本数
                                       shuffle=True,
                                       num_workers=0)

    val_dataloader = data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=0)

    return train_dataloader, val_dataloader


# 训练模型
def train_model_process(model, train_dataloader, val_dataloader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优化器，用以自动调整模型参数。Adam移动平均权值，抑制梯度消失或爆炸，加速梯度下降
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 交叉熵损失函数，用以分类（回归一般用均方差）
    criterion = nn.CrossEntropyLoss()

    # 模型放入设备中
    model = model.to(device)

    # 复制当前模型参数，用于优化模型
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # 初始化参数
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        # 取一批次的数据，进行训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 开启训练模式
            model.train()
            # 前向传播，输入为一个批次，输出为这个批次对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标，即通过softmax，通过这里就可以得到结果了
            pre_lab = torch.argmax(output, dim=1)
            # 计算损失值，由于交叉熵损失函数中内置了softmax，因此用output来算
            loss = criterion(output, b_y)
            # 初始化梯度
            optimizer.zero_grad()
            # 反向传播，此时生成新的梯度
            loss.backward()
            # 根据反向传播的梯度信息来更新网络参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 计算准确度，预测正确时+1
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 验证
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 开启评估模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        # 计算并保持每轮次loss和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 打印该轮次的值，-1代表最新的一个
        print('{} Train Loss: {:.4f} Train ACC:{:.4f}'.format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val ACC:{:.4f}'.format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        # 优化模型，寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练时间
        time_use = time.time() - since
        print("Using time:{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 保存最优参数
    torch.save(best_model_wts, 'output_models/%s_best_model.pth' % model.model_name)
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


# 画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(8, 5))

    # 初始化主坐标系
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制loss曲线（左侧轴）
    loss_train = ax1.plot(train_process["epoch"], train_process.train_loss_all,
                          color='red', linestyle='--', marker='D',
                          markersize=3, label="train loss")
    loss_val = ax1.plot(train_process["epoch"], train_process.val_loss_all,
                        color='blue', linestyle='--', marker='D',
                        markersize=3, label="val loss")

    # 绘制acc曲线（右侧轴）
    acc_train = ax2.plot(train_process["epoch"], train_process.train_acc_all,
                         color='red', linestyle='-', marker='D',
                         markersize=3, label="train acc")
    acc_val = ax2.plot(train_process["epoch"], train_process.val_acc_all,
                       color='blue', linestyle='-', marker='D',
                       markersize=3, label="val acc")

    # 坐标轴设置
    ax1.set_xlabel("Epochs", fontsize=8)
    ax1.set_ylabel("Loss", color='#2C3E50', fontsize=8)
    ax2.set_ylabel("Accuracy", color='#2C3E50', fontsize=8)

    # 强制x轴整数刻度
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 统一视觉样式
    for ax in [ax1, ax2]:
        ax.tick_params(axis='y', labelcolor='#2C3E50')
        ax.spines['top'].set_visible(False)
        ax.spines[['right', 'left', 'bottom']].set_color('#BDC3C7')

    # 合并图例
    all_lines = loss_train + loss_val + acc_train + acc_val
    ax1.legend(all_lines, [l.get_label() for l in all_lines],
               loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, -0.15),
               frameon=True, framealpha=0.9)

    plt.title("Loss and ACC of %s" % model.model_name,
              fontsize=14, pad=20, color='#2C3E50')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('./images/LossAndACC_%s.png' % model.model_name)
    plt.show()


if __name__ == "__main__":
    choose_model = int(input('1: LeNet; 2: AlexNet; 3:VGGNet16; 4: GoogLeNet; else: ResNet18\n'
                             'Select Model:'))

    # 模型实例化
    if choose_model == 1:
        model = LeNet()
    elif choose_model == 2:
        model = AlexNet()
    elif choose_model == 3:
        model = VGG16()
    elif choose_model == 4:
        model = GoogLeNet()
    else:
        model = ResNet18()
    print("Selected Model = %s" % model.model_name)

    # 加载数据集
    train_dataloader, val_dataloader = train_val_data_process()
    # 开始训练
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20, lr=0.001)
    # 画图
    matplot_acc_loss(train_process)
