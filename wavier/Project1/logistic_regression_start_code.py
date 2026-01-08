import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import csv
import random

def feature_normalization(train, test):
    train_min=train.min(axis=0)
    train_max=train.max(axis=0)
    train_normalized=((train-train_min)/(train_max-train_min))
    test_min = test.min(axis=0)
    test_max = test.max(axis=0)
    test_normalized = ((test - test_min) / (test_max - test_min))
    return train_normalized, test_normalized

# 加载数据

data = pd.read_csv('D:/Desktop/大三下/机器学习/HW1/ml2024_hw1 3/data/bank-full.csv', delimiter=',')


# 数据预处理

label_encoder = LabelEncoder()
data['job'] = label_encoder.fit_transform(data['job'])
data['marital'] = label_encoder.fit_transform(data['marital'])
data['education'] = label_encoder.fit_transform(data['education'])
data['default'] = label_encoder.fit_transform(data['default'])
data['housing'] = label_encoder.fit_transform(data['housing'])
data['loan'] = label_encoder.fit_transform(data['loan'])
data['contact'] = label_encoder.fit_transform(data['contact'])
data['month'] = label_encoder.fit_transform(data['month'])
data['poutcome'] = label_encoder.fit_transform(data['poutcome'])
data['y'] = label_encoder.fit_transform(data['y'])
#删除无效值
data_dropna = data.dropna()

X = data.values[:, :-1]
y = data.values[:, -1]

print('Split into Train and Test')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)
X_train, X_test=feature_normalization(X_train, X_test)
y_train, y_test=feature_normalization(y_train, y_test)

# TODO 4.1.1 将非数值字段转换为为数值，进行缺失值处理


# 划分数据集
# TODO 4.1.1 特征标准化


# 训练逻辑斯特回归模型
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
# TODO 4.1.2 训练逻辑斯特回归模型，打印模型在测试集上的准确率以及混淆矩阵

# 可视化特征权重
weights = model.coef_
weights_new=weights[0]
num_features=X_train.shape[1]
features=range(num_features)
plt.figure(figsize=(8, 4))
plt.bar(features, weights_new, color='blue', label='effect')
plt.show()
#绘画柱形图
# TODO 4.1.3 提取特征权重并进行分析

# 手动实现逻辑斯特回归，并进行模型训练以及模型评估
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def gradient(X, y, theta):
    grad=np.dot((sigmoid(np.dot(X,theta))-y),X)
    return grad

def stochastic_grad_descent(X_train, y_train, alpha=0.1, num_iter=1000, batch_size=100):
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    for i in range(num_iter):
        sX=random.sample(range(X_train.shape[0]),batch_size)
        stochastic_X_train=X_train[sX,:]
        sY=sX
        stochastic_y_train=y_train[sY]
        grad = gradient(stochastic_X_train, stochastic_y_train, theta)
        theta = theta - alpha * grad
        theta_hist[i + 1:] = theta

    return theta

theta=stochastic_grad_descent(X_train, y_train, alpha=0.1, num_iter=1000, batch_size=1)
y_pred_prob = np.dot(X_test, theta)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
# TODO 4.1.5




