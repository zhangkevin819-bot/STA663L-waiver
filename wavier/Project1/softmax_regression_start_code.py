from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO 4.2.1 特征归一化与模型训练
def feature_normalization(train, test):
    train_min=train.min(axis=0)
    train_max=train.max(axis=0)
    train_normalized=((train-train_min)/(train_max-train_min))
    test_min = test.min(axis=0)
    test_max = test.max(axis=0)
    test_normalized = ((test - test_min) / (test_max - test_min))
    return train_normalized, test_normalized

X_train, X_test=feature_normalization(X_train, X_test)



#进行softmax训练
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)



# TODO 4.2.2 你可以任意选取不同的特征用于特征可视化
feature_x_index = 0  # 第一个特征的索引
feature_y_index = 3  # 第二个特征的索引
X_train[:,1]=X_train[:,1].mean()
X_train[:,2]=X_train[:,2].mean()

# 创建网格来绘制决策边界，你将得到两个形状为[k, k]的网格
x_min, x_max = X_train[:, feature_x_index].min() - 0.1, X_train[:, feature_x_index].max() + 0.1
y_min, y_max = X_train[:, feature_y_index].min() - 0.1, X_train[:, feature_y_index].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))
num_k=(xx.ravel()).shape[0]
X_train_new=np.zeros((num_k,4))
X_train_new[:,0]=xx.ravel()
X_train_new[:,3]=yy.ravel()
X_train_new[:,1]=X_train[:,1].mean()
X_train_new[:,2]=X_train[:,2].mean()
Z0 = model.predict(X_train_new)
Z=Z0.reshape(xx.shape)

# 进行预测并绘制决策边界
# TODO 4.2.2
# 设网格长度为k，则创建好的两个网格应当包含k*k个数据点，相当于你已经有了k*k个样本的其中两个特征，
# 你需要为他们填充剩余两个特征(可以使用训练集样本对应特征的均值或其他统计量进行填充)。填充完
# 毕后，你应当将该特征矩阵转变为[k*k, 4]这样的形状，方便模型进行预测



# 绘制决策边界, Z是使用模型对k*k个点进行预测得到的样本标签，其形状需要被转换为[k, k]
plt.contourf(xx, yy, Z, alpha=0.8)

# 绘制数据点
plt.scatter(X_train[:, feature_x_index], X_train[:, feature_y_index], c=y_train, edgecolors='k')
plt.xlabel(iris.feature_names[feature_x_index])
plt.ylabel(iris.feature_names[feature_y_index])
plt.title('Decision Boundary with Softmax Regression')
plt.show()

# TODO 4.2.3
#基函数有2维时
poly = PolynomialFeatures(degree=2)
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
pipeline = Pipeline([
    ('polynomial_features', poly),
    ('logistic_regression', softmax_reg)
])
pipeline.fit(X_train, y_train)
Z0 = pipeline.predict(X_train_new)
Z =Z0.reshape(xx.shape)
# 绘制决策边界, Z是使用模型对k*k个点进行预测得到的样本标签，其形状需要被转换为[k, k]
plt.contourf(xx, yy, Z, alpha=0.8)
# 绘制数据点
plt.scatter(X_train[:, feature_x_index], X_train[:, feature_y_index], c=y_train, edgecolors='k')
plt.xlabel(iris.feature_names[feature_x_index])
plt.ylabel(iris.feature_names[feature_y_index])
plt.title('Decision Boundary with Softmax Regression, degree=2')
plt.show()

#重复操作，基函数有3维时
poly = PolynomialFeatures(degree=3)
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
pipeline = Pipeline([
    ('polynomial_features', poly),
    ('logistic_regression', softmax_reg)
])
pipeline.fit(X_train, y_train)
Z0 = pipeline.predict(X_train_new)
Z =Z0.reshape(xx.shape)
# 绘制决策边界, Z是使用模型对k*k个点进行预测得到的样本标签，其形状需要被转换为[k, k]
plt.contourf(xx, yy, Z, alpha=0.8)
# 绘制数据点
plt.scatter(X_train[:, feature_x_index], X_train[:, feature_y_index], c=y_train, edgecolors='k')
plt.xlabel(iris.feature_names[feature_x_index])
plt.ylabel(iris.feature_names[feature_y_index])
plt.title('Decision Boundary with Softmax Regression, degree=3')
plt.show()

#重复操作，基函数有4维时
poly = PolynomialFeatures(degree=4)
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
pipeline = Pipeline([
    ('polynomial_features', poly),
    ('logistic_regression', softmax_reg)
])
pipeline.fit(X_train, y_train)
Z0 = pipeline.predict(X_train_new)
Z =Z0.reshape(xx.shape)
# 绘制决策边界, Z是使用模型对k*k个点进行预测得到的样本标签，其形状需要被转换为[k, k]
plt.contourf(xx, yy, Z, alpha=0.8)
# 绘制数据点
plt.scatter(X_train[:, feature_x_index], X_train[:, feature_y_index], c=y_train, edgecolors='k')
plt.xlabel(iris.feature_names[feature_x_index])
plt.ylabel(iris.feature_names[feature_y_index])
plt.title('Decision Boundary with Softmax Regression, degree=4')
plt.show()

#重复操作，基函数有5维时
poly = PolynomialFeatures(degree=5)
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
pipeline = Pipeline([
    ('polynomial_features', poly),
    ('logistic_regression', softmax_reg)
])
pipeline.fit(X_train, y_train)
Z0 = pipeline.predict(X_train_new)
Z =Z0.reshape(xx.shape)
# 绘制决策边界, Z是使用模型对k*k个点进行预测得到的样本标签，其形状需要被转换为[k, k]
plt.contourf(xx, yy, Z, alpha=0.8)
# 绘制数据点
plt.scatter(X_train[:, feature_x_index], X_train[:, feature_y_index], c=y_train, edgecolors='k')
plt.xlabel(iris.feature_names[feature_x_index])
plt.ylabel(iris.feature_names[feature_y_index])
plt.title('Decision Boundary with Softmax Regression, degree=5')
plt.show()
# 为原始数据增加多项式特征，重复上述步骤以绘制决策边界