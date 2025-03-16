import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model.LeNet import LeNet  # 导入模型
from model.AlexNet import AlexNet
from model.VGGNet16 import VGG16
from model.GoogLeNet import GoogLeNet
from model.ResNet18 import ResNet18



def test_data_process():
    test_data = FashionMNIST(root='./dataset',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=model.size_input),
                                                           transforms.ToTensor()]),
                             download=True)

    test_dataloader = data.DataLoader(dataset=test_data,
                                      batch_size=1,  # 一批次样本数
                                      shuffle=True,
                                      num_workers=0)

    return test_dataloader


def test_model_process_acc(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 梯度置零，不计算梯度，只进行前向传播
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置为评估模式
            model.eval()
            # 前向传播
            output = model(b_x)
            # 查找每一行中最大行标
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == b_y.data)
            test_num += b_x.size(0)

    # 计算准确率
    test_acc = test_corrects.double().item() / test_num
    print("test accuracy: ", test_acc)


def test_model_process_label(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            # 获取数值
            result = pre_lab.item()
            label = b_y.item()
            print("predicted value:", classes[result], "------ actual value:", classes[label])


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

    # 加载模型
    model.load_state_dict(torch.load('./output_model/%s_best_model.pth' % model.model_name))
    # 加载数据
    test_dataloader = test_data_process()
    # 输出标签
    test_model_process_label(model, test_dataloader)
    # 输出准确率
    test_model_process_acc(model, test_dataloader)
