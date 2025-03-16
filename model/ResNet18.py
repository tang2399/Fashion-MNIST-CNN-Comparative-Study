import torch
from torch import nn
from torchsummary import summary


# 残差块
class Residual(nn.Module):
    def __init__(self, inputChannels, numChannels, use_1Conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()

        self.c1 = nn.Conv2d(in_channels=inputChannels, out_channels=numChannels,
                            kernel_size=3, padding=1, stride=strides)
        self.bn2 = nn.BatchNorm2d(numChannels)
        self.c3 = nn.Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(numChannels)

        # 第三个残差块使用了1x1卷积
        if use_1Conv:
            self.c5 = nn.Conv2d(in_channels=inputChannels, out_channels=numChannels,
                                kernel_size=1, stride=strides)
        else:
            self.c5 = None

    def forward(self, x):
        y = self.ReLU(self.bn2(self.c1(x)))
        y = self.bn4(self.c3(y))

        if self.c5:
            x = self.c5(x)

        return self.ReLU(x + y)


class ResNet18(nn.Module):
    def __init__(self, Residual=Residual):
        super(ResNet18, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64),
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1Conv=True, strides=2),
            Residual(128, 128),
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1Conv=True, strides=2),
            Residual(256, 256),
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1Conv=True, strides=2),
            Residual(512, 512),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        self.size_input = 224
        self.model_name = "ResNet18"

    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        x = self.b6(self.b5(self.b4(x)))

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(Residual).to(device)
    print(summary(model, (1, 224, 224)))
