import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)()
    """
        TODO
        参考2.1.1，定义的多层感知机的网络结构，大约4行代码。

        可能会用到的算子：
        - nn.Linear 线性层
        - nn.Softmax Softmax激活函数

        这些算子的具体使用方法，参加Pytorch官方文档，https://pytorch.org/docs/stable/index.html
        """

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z1 = self.layer1(x)
        h1 = torch.relu(z1)
        z2 = self.layer2(h1)
        h2 = torch.relu(z2)
        z3 = self.layer3(h2)
        y_hat = self.softmax(z3)
        return y_hat
        """
        TODO
        实现已定义的卷积神经网络的前向传播过程，大约6行代码。
        输入：x  (b,28,28)的灰度图像，b为批大小。
        """
        pass
