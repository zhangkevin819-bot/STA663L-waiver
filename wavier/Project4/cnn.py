import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)()
        """
        TODO
        参考2.2.1，定义卷积神经网络的网络结构，大约6行代码。

        可能会用到的算子：
        - nn.Conv2d 二维卷积算子
        - nn.MaxPool2d 最大池化算子
        - nn.Linear 线性层
        - nn.Softmax Softmax激活函数

        这些算子的具体使用方法，参加Pytorch官方文档，https://pytorch.org/docs/stable/index.html
        """

        

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
        """
        TODO
        实现已定义的卷积神经网络的前向传播过程，大约七行代码。
        输入：x  (b,28,28)的灰度图像，b为批大小。
        """
        pass
        