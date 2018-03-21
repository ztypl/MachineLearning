# coding : utf-8
# create by ztypl on 2017/10/27

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('buy.csv', encoding='gbk', index_col=0)
df = df.dropna(axis=1)
df = df.iloc[:,1:]

names = df.index
data = df.values


N = 8

km = KMeans(n_clusters=N)

label = km.fit_predict(data)
expenses = np.sum(km.cluster_centers_,axis=1)

result = []

for i in range(N):
    result.append([df.index[label==i].values.tolist(), expenses[i]])

result.sort(key=lambda x:x[1], reverse=True)

for index, val in enumerate(result):
    print("cluster %d: expenses %.2f" % (index, val[1]))
    print("\t%s" % val[0])
