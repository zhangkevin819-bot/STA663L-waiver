import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# 读取数据
file_path_train = 'D:/Desktop/大三下/机器学习/HW1/ml2024_hw1 3/data/real_estate_train.xlsx'
data_train = pd.read_excel(file_path_train)
file_path_test = 'D:/Desktop/大三下/机器学习/HW1/ml2024_hw1 3/data/real_estate_test.xlsx'
data_test = pd.read_excel(file_path_test)


#3.1
#定义特征工程中处理时间变量的函数
def parse_date(decimal_date):
    # 分割年份和小数部分
    decimal_string = str(decimal_date)
    year_part, fraction_part = decimal_string.split('.')
    year = int(year_part)
    # 将小数部分转换为天数
    # 一年按365天计算，考虑闰年
    is_leap = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    days = int(float("0." + fraction_part) * days_in_year)
    # 计算具体日期
    date = datetime(year, 1, 1) + timedelta(days=days)
    return date



#创造关于年月日变量的列
parsed_dates1 = [parse_date(date) for date in data_train['X1 transaction date']]
df = pd.DataFrame(parsed_dates1, columns=['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

#将年月日变量的列插入到训练集中
data_train_new1=data_train.drop('X1 transaction date', axis=1)
data_train_new2=data_train.drop('No', axis=1)
extend_train=pd.concat([df, data_train_new2], axis=1)
final_train=extend_train.drop('Date', axis=1)

#创造关于年月日变量的列
parsed_dates2 = [parse_date(date) for date in data_test['X1 transaction date']]
dk = pd.DataFrame(parsed_dates2, columns=['Date'])
dk['Year'] = dk['Date'].dt.year
dk['Month'] = dk['Date'].dt.month
dk['Day'] = dk['Date'].dt.day

#将年月日变量的列插入到训练集中
data_test_new1=data_test.drop('X1 transaction date', axis=1)
data_test_new2=data_test.drop('No', axis=1)
extend_test=pd.concat([dk, data_test_new2], axis=1)
final_test=extend_test.drop('Date', axis=1)

def feature_normalization(train, test):
    train_min=train.min(axis=0)
    train_max=train.max(axis=0)
    train_normalized=((train-train_min)/(train_max-train_min))
    test_min = test.min(axis=0)
    test_max = test.max(axis=0)
    test_normalized = ((test - test_min) / (test_max - test_min))
    return train_normalized, test_normalized

#进行归一化并获取要使用的训练集与验证集
train_normalized, test_normalized = feature_normalization(final_train, final_test)
X_train = train_normalized.values[:, :-1]
y_train = train_normalized.values[:, -1]
X_test = test_normalized.values[:, :-1]
y_test = test_normalized.values[:, -1]
# 你需要进行特征工程，使用线性回归或其他适用的模型进行房价预测，使用交叉验证选择合适的超参数，并在测试集上进行模型评估。
#3.2
#定义相对L2误差
def relative_L2_error(y_true, y_pred):
    return np.linalg.norm((y_true - y_pred) ** 2)/np.linalg.norm(y_true**2)

#先用岭回归
ridge = Ridge()
# 参数网格
param_grid = {'alpha': np.logspace(-4, 4, 20)}

# 创建 GridSearchCV 对象，进行岭回归下用交叉验证来选择超参数
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))


#计算岭回归下系数与相对L2误差
ridge = Ridge(alpha=grid_search.best_params_['alpha'])
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
error=relative_L2_error(y_test, y_pred)
coefficients=ridge.coef_
print("ridge error:",error)
print("coefficients:",coefficients)


#再用lasso回归
lasso = Lasso()

# 参数网格
param_grid = {'alpha': np.logspace(-4, 4, 20)}

# 创建 GridSearchCV 对象,进行lasso回归下用交叉验证来选择超参数
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

#计算lasso回归下系数与相对L2误差
lasso = Lasso(alpha=grid_search.best_params_['alpha'])
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
error=relative_L2_error(y_test, y_pred)
coefficients=lasso.coef_
print("lasso error:",error)
print("coefficients:",coefficients)


