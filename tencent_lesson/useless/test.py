# coding : utf-8
# create by ztypl on 2017/9/6

import pickle

x = pickle.load(open('data.pkl', 'rb'))
data = x['data']
data_new = []
for i in data:
    data_new.append(i[0])
x['data'] = data_new
pickle.dump(x, open('data.pkl', 'wb'))


