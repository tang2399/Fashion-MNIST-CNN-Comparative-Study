import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sig = nn.Sigmoid()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)  # 第一层卷积层
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 第一层池化层

        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()  # 平展层
        self.f5 = nn.Linear(in_features=400, out_features=120)  # 全连接层
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

        self.size_input = 28
        self.model_name = "LeNet"

    def forward(self, x):  # x为模型的输入
        x = self.s2(self.sig(self.c1(x)))
        x = self.s4(self.sig(self.c3(x)))
        x = self.flatten(x)
        x = self.sig(self.f5(x))
        x = self.sig(self.f6(x))
        x = self.f7(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))
