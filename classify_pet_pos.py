# 导入相关包
import numpy as np
import pandas as pd
# 导入数据集
from sklearn.datasets import load_iris
# 导入模型
from sklearn.tree import DecisionTreeClassifier
# 数据分割包
from sklearn.model_selection import train_test_split
# 评价包
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
# 规范化数据包
from sklearn import preprocessing

data = pd.read_csv('./pet_data.csv')
#data.shape

x = data.iloc[:, :4]
y = data.iloc[:, 4]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4)
# 数据规范化
ss = preprocessing.StandardScaler()
ss_train_x = ss.fit_transform(train_x)
ss_test_x = ss.fit_transform(test_x)

# 使用决策树进行分类
dtc = DecisionTreeClassifier()
dtc.fit(ss_train_x, train_y)
predict_y = dtc.predict(ss_test_x)
# 评分
print("mean_squared_error:", mean_squared_error(test_y, predict_y))
print('accuracy_score：', accuracy_score(test_y, predict_y))

#使用SVM进行分类
from sklearn.svm import SVC
svc = SVC()
svc.fit(ss_train_x, train_y)
predict_y = svc.predict(ss_test_x)
# 评分
print("mean_squared_error:", mean_squared_error(test_y, predict_y))
print('accuracy_score：', accuracy_score(test_y, predict_y))

#使用KNN进行分类
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(ss_train_x, train_y)
predict_y = knn.predict(ss_test_x)
# 评分
print("mean_squared_error:", mean_squared_error(test_y, predict_y))
print('accuracy_score：', accuracy_score(test_y, predict_y))

#使用boost进行分类
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(ss_train_x, train_y)
predict_y = ada.predict(ss_test_x)
# 评分
print("mean_squared_error:", mean_squared_error(test_y, predict_y))
print('accuracy_score：', accuracy_score(test_y, predict_y))

