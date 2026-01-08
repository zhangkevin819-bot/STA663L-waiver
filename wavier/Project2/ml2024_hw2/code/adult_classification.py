import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# 加载数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                'native-country', 'income']
data = pd.read_csv(url, names=column_names, na_values='?', skipinitialspace=True)

# TODO 4.1 数据预处理
data=data.dropna()
m = LabelEncoder()
for column in column_names:
    data[column] = m.fit_transform(data[column])

numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 划分数据集
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO 4.2 使用决策树模型进行预测，并进行模型评估
dt_clf1 = DecisionTreeClassifier(random_state=42,max_depth=5)
dt_clf1.fit(X_train, y_train)
y_pred_dt = dt_clf1.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy}")

dt_clf2 = DecisionTreeClassifier(random_state=42,max_depth=10)
dt_clf2.fit(X_train, y_train)
y_pred_dt = dt_clf2.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy}")

dt_clf3 = DecisionTreeClassifier(random_state=42,max_depth=15)
dt_clf3.fit(X_train, y_train)
y_pred_dt = dt_clf3.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy}")

# TODO 4.3 使用随机森林模型进行预测，并进行模型评估
rf_clf1 = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=5,max_features=10)
rf_clf1.fit(X_train, y_train)
y_pred_rf = rf_clf1.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

rf_clf1 = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=5,max_features=10)
rf_clf1.fit(X_train, y_train)
y_pred_rf = rf_clf1.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

rf_clf2 = RandomForestClassifier(n_estimators=50, random_state=42,max_depth=5,max_features=10)
rf_clf2.fit(X_train, y_train)
y_pred_rf = rf_clf2.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

rf_clf3 = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10,max_features=10)
rf_clf3.fit(X_train, y_train)
y_pred_rf = rf_clf3.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

rf_clf4 = RandomForestClassifier(n_estimators=50, random_state=42,max_depth=10,max_features=10)
rf_clf4.fit(X_train, y_train)
y_pred_rf = rf_clf4.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

rf_clf5 = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10,max_features=5)
rf_clf5.fit(X_train, y_train)
y_pred_rf = rf_clf5.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")


# TODO 4.4 可视化得到的决策树，分析各个字段在模型预测过程中起的作用
plot_tree(dt_clf1, filled=True)
plt.show()

plot_tree(dt_clf2, filled=True)
plt.show()

plot_tree(dt_clf3, filled=True)
plt.show()

# TODO 4.5 使用xgboost进行预测，通过调整参数优化模型在验证集上的表现
xgb_clf = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy}")

xgb_clf = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=10, random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy}")

xgb_clf = XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=10, random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy}")

xgb_clf = XGBClassifier(learning_rate=0.5, n_estimators=100, max_depth=10, random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy}")