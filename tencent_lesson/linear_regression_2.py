# coding : utf-8
# create by ztypl on 2017/9/7

"""Ten baseline variables, age, sex, body mass index,
average blood pressure, and six blood serum measurements
were obtained for each of n = 442 diabetes patients,
as well as the response of interest, a quantitative
measure of disease progression one year after baseline."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# 读取自带的diabete数据集
diabetes = datasets.load_diabetes()


# 使用其中的一个feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# 将数据集分割成training set和test set
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标（y值）分割成training set和test set
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 使用线性回归
regr = linear_model.LinearRegression()

# 进行training set和test set的fit，即是训练的过程
regr.fit(diabetes_X_train, diabetes_y_train)

# 打印出相关系数和截距等信息
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# 使用pyplot画图
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red', marker='x')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()