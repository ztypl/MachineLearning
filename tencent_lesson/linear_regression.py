# coding : utf-8
# create by ztypl on 2017/9/7

from sklearn import linear_model
import matplotlib.pyplot as plt

x = [[1], [2], [3], [4], [5]]
y = [2.8, 3.4, 3.7, 3.8, 4.2]


model = linear_model.LinearRegression()
model.fit(x, y)

plt.scatter(x, y,  color='black')
plt.plot(x, model.predict(x), color='blue',
         linewidth=3)

print(model.predict(6))

plt.show()