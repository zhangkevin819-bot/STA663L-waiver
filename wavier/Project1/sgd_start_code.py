import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
import random
import csv
with open('D:/Desktop/大三下/机器学习/HW1/__MACOSX/ml2024_hw1 3/data/._sgd_data.csv', newline='') as file:
    reader = csv.reader(file)

def feature_normalization(train, test):
    train_min=train.min(axis=0)
    train_max=train.max(axis=0)
    train_normalized=((train-train_min)/(train_max-train_min))
    test_min = test.min(axis=0)
    test_max = test.max(axis=0)
    test_normalized = ((test - test_min) / (test_max - test_min))
    return train_normalized, test_normalized
    """将训练集中的所有特征值映射至[0,1]，对验证集上的每个特征也需要使用相同的仿射变换

    Args：
        train - 训练集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        test - 测试集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
    Return：
        train_normalized - 归一化后的训练集
        test_normalized - 标准化后的测试集

    """
    # TODO 2.1


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    num_instances = X.shape[0]
    loss = np.linalg.norm(np.dot(X,theta) - y)/ num_instances  + lambda_reg*np.linalg.norm(theta)
    return loss
    """
    给定一组 X, y, theta，计算用 X*theta 预测 y 的岭回归损失函数

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小 (num_features)
        lambda_reg - 正则化系数

    Return：
        loss - 损失函数，标量
    """
    # TODO 2.2.2


def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    num_instances = X.shape[0]
    grad = 2*np.dot((np.dot(X,theta) - y),X)/num_instances + 2*lambda_reg*theta
    return grad
    """
    计算岭回归损失函数的梯度

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数

    返回：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.2.4



def batch_grad_descent(X, y, lambda_reg, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    全批量梯度下降算法

    Args：
        X - 特征向量， 数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        check_gradient - 更新时是否检查梯度

    Return：
        theta_hist - 存储迭代中参数向量的历史，大小为 (num_iter+1, num_features) 的二维 numpy 数组
        loss_hist - 全批量损失函数的历史，大小为 (num_iter) 的一维 numpy 数组
    """
    num_instances,num_features= X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    for i in range(num_iter):
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha*grad
        loss_hist[i] = compute_regularized_square_loss(X, y, theta, lambda_reg)
        theta_hist[i+1:] = theta

    return theta_hist,loss_hist



    # TODO 2.3.3


def compute_current_alpha(alpha, iter):
    """
    梯度下降步长策略，可自行扩展支持更多策略

    参数：
        alpha - 字符串或浮点数。梯度下降步长
                注意：在 SGD 中，使用固定步长并不总是一个好主意。通常设置为 1/sqrt(t) 或 1/t
                如果 alpha 是一个浮点数，那么每次迭代的步长都是 alpha。
                如果 alpha == "0.05/sqrt(t)", alpha = 0.05/sqrt(t)
                如果 alpha == "0.05/t", alpha = 0.05/t
        iter - 当前迭代次数（初始为1）

    返回：
        current_alpha - 当前采取的步长
    """
    assert isinstance(alpha, float) or (isinstance(alpha, str) and (alpha == '0.05/sqrt(t)' or alpha == '0.05/t'))
    if isinstance(alpha, float):
        current_alpha = alpha
    elif alpha == '0.05/sqrt(t)':
        current_alpha = 0.05 / np.sqrt(iter)
    elif alpha == '0.05/t':
        current_alpha = 0.05 / iter
    return current_alpha


def stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1):
    """
    随机梯度下降，并随着训练过程在验证集上验证

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_test - 验证集特征向量，数组大小 (num_instances, num_features)
        y_test - 验证集标签向量，数组大小 (num_instances)
                 注意：在 SGD 中，小批量的训练损失函数噪声较大，难以清晰反应模型收敛情况，可以通过验证集上的全批量损失来判断
        alpha - 字符串或浮点数。梯度下降步长，可自行调整为默认值以外的值
                注意：在 SGD 中，使用固定步长并不总是一个好主意。通常设置为 alpha_0/sqrt(t) 或 alpha_0/t
                如果 alpha 是一个浮点数，那么每次迭代的步长都是 alpha。
                如果 alpha == "0.05/sqrt(t)", alpha = 0.05/sqrt(t)
                如果 alpha == "0.05/t", alpha = 0.05/t
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量正则化损失函数的历史，数组大小(num_iter)
        validation hist - 验证集上全批量均方误差（不带正则化项）的历史，数组大小(num_iter)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist
    for i in range(num_iter):
        sX=random.sample(range(X_train.shape[0]),batch_size)
        stochastic_X_train=X_train[sX,:]
        sY=sX
        stochastic_y_train=y_train[sY]
        grad = compute_regularized_square_loss_gradient(stochastic_X_train, stochastic_y_train, theta, lambda_reg)
        theta = theta - alpha * grad
        loss_hist[i] = compute_regularized_square_loss(stochastic_X_train, stochastic_y_train, theta, lambda_reg)
        theta_hist[i + 1:] = theta
        validation=np.linalg.norm(np.dot(X_test,theta) - y_test)
        validation_hist[i]=validation

    return theta_hist, loss_hist, validation_hist
    # TODO 2.4.2
    
    
def lasso_regression(X_train, y_train, lambda_reg):
    """
    调整不同的正则化系数，分析模型参数的稀疏性

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_test - 验证集特征向量，数组大小 (num_instances, num_features)
        y_test - 验证集标签向量，数组大小 (num_instances)
                 注意：在 SGD 中，小批量的训练损失函数噪声较大，难以清晰反应模型收敛情况，可以通过验证集上的全批量损失来判断
        lambda_reg - 正则化系数，可自行调整为默认值以外的值

    返回：
        theta - lasso回归的参数向量，可以通过lasso模型的coef_参数直接获取
    """
    lasso = Lasso(alpha=lambda_reg)
    lasso.fit(X_train, y_train)
    theta=lasso.coef_

    return theta

    # TODO 2.4.5


def main():
    # 加载数据集
    print('loading the dataset')

    df = pd.read_csv('D:/Desktop/大三下/机器学习/HW1/ml2024_hw1 3/data/sgd_data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)
    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    y_train, y_test = feature_normalization(y_train, y_test)
    num_features=X_train.shape[1]
    print("Num Features: ", num_features)

    # 选取不同批，记录训练曲线
    theta_hist, loss_hist = batch_grad_descent(X_train, y_train, lambda_reg=0.001, alpha=0.5, num_iter=1000)
    plt.figure(figsize=(10,5))
    plt.plot(loss_hist)
    plt.title("Loss with alpha=0.5")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    theta_hist, loss_hist = batch_grad_descent(X_train, y_train, lambda_reg=0.001, alpha=0.1, num_iter=1000)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist)
    plt.title("Loss with alpha=0.1")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


    theta_hist, loss_hist = batch_grad_descent(X_train, y_train, lambda_reg=0.001, alpha=0.05, num_iter=1000)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist)
    plt.title("Loss with alpha=0.05")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


    theta_hist, loss_hist = batch_grad_descent(X_train, y_train, lambda_reg=0.001, alpha=0.01, num_iter=1000)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist)
    plt.title("Loss with alpha=0.01")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    theta_hist, loss_hist, validation_hist=stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg=0, alpha=0.01, num_iter=1000, batch_size=1)
    plt.figure(figsize=(10, 5))
    plt.plot(validation_hist)
    plt.title("Loss with batch=1")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    theta_hist, loss_hist, validation_hist = stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg=0, alpha=0.01, num_iter=1000, batch_size=5)
    plt.figure(figsize=(10, 5))
    plt.plot(validation_hist)
    plt.title("Loss with batch=5")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    theta_hist, loss_hist, validation_hist = stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg=0,
                                                                     alpha=0.01, num_iter=1000, batch_size=10)
    plt.figure(figsize=(10, 5))
    plt.plot(validation_hist)
    plt.title("Loss with batch=10")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    theta_hist, loss_hist, validation_hist = stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg=0,
                                                                     alpha=0.01, num_iter=1000, batch_size=20)
    plt.figure(figsize=(10, 5))
    plt.plot(validation_hist)
    plt.title("Loss with batch=20")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # 选取不同步长，计算均方误差
    validation_hist1 = np.zeros(7)
    i=0
    values=[1e-07, 1e-05, 1e-03, 0.01, 1, 10, 100]
    for lambda_reg in values:
        theta_hist, loss_hist, validation_hist = stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg,
                                                                         alpha=0.01, num_iter=1000, batch_size=20)
        validation_hist1[i] = validation_hist[999]
        i=i+1

    table={
        'lambda_reg': values,
        'validation_hist': validation_hist1
    }

    df = pd.DataFrame(table)
    print(df)


    #选取不同正则化系数来绘制条形图以估计稀疏性
    num_features=X_train.shape[1]
    features=range(num_features)
    theta=lasso_regression(X_train, y_train, lambda_reg=0.001)
    plt.figure(figsize=(8, 4))
    plt.bar(features, theta, color='blue', label='alpha=0.001')
    plt.show()
    theta = lasso_regression(X_train, y_train, lambda_reg=0.01)
    plt.figure(figsize=(8, 4))
    plt.bar(features, theta, color='blue', label='alpha=0.01')
    plt.show()
    theta = lasso_regression(X_train, y_train, lambda_reg=0.05)
    plt.figure(figsize=(8, 4))
    plt.bar(features, theta, color='blue', label='alpha=0.05')
    plt.show()
    theta = lasso_regression(X_train, y_train, lambda_reg=0.1)
    plt.figure(figsize=(8, 4))
    plt.bar(features, theta, color='blue', label='alpha=0.1')
    plt.show()
    theta = lasso_regression(X_train, y_train, lambda_reg=0.5)
    plt.figure(figsize=(8, 4))
    plt.bar(features, theta, color='blue', label='alpha=0.5')
    plt.show()





    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 增加偏置项
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # 增加偏置项

    # TODO


if __name__ == "__main__":
    main()
