import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

from models import ViT,CNN, MLP



# 超参数
#batch_size = 64
#batch_size = 32
batch_size = 100
#learning_rate = 0.0005
#learning_rate = 0.0001
learning_rate = 0.001
#epochs = 5
epochs = 10
#epochs = 15

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载训练集和测试集，默认路径为 './data'
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 选择一个模型，ViT 或者 CNN
#model = CNN()
model = ViT()
#model = MLP()
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 余弦学习率调节器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 训练模型
def train(model, device, train_loader, optimizer, scheduler, epoch):
    correct = 0
    model.train()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad() # 清空优化器中缓存的梯度。
        loss.backward() # 反向传播
        optimizer.step() # 优化器更新参数
        pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
        correct += pred.eq(target.view_as(pred)).sum().item()  # 计算预测的准确率

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / batch_size:.6f}')
    scheduler.step()
    end_time = time.time()
    print(f'Epoch {epoch} completed in {end_time - start_time:.2f} seconds')
    print(f'\nTrain set:Accuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.0f}%)\n')
    return loss.item() / batch_size

# 测试模型
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item() # 计算预测的准确率

    test_loss /= len(test_loader.dataset)
    average_test_loss=test_loss / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return average_test_loss
# 设置设备（CPU或GPU）
device = torch.device("cpu")
model.to(device)

train_losses=[]
test_losses=[]
# 运行训练和测试
for epoch in range(1, epochs + 1):
    average_train_loss=train(model, device, train_loader, optimizer, scheduler, epoch)
    average_test_loss=test(model, device, test_loader)
    train_losses.append(average_train_loss)
    test_losses.append(average_test_loss)



plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(test_losses, label='Testing Loss')
plt.title('Testing Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()