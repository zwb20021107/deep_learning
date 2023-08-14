# -*- coding: utf-8 -*-
# @Time : 2023-08-14 13:19
# @Author : ZuoWenBin


# 画图
import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import  utils


iris = sns.load_dataset("iris")

# 花瓣和花萼的数据
X = iris.values[:, :4]
# 标签值
y = iris.values[:, 4]
# # 展示数据前五
# print(iris.head())
#
#
# sns.pairplot(iris, hue="species")

# 进行热编码
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return utils.to_categorical(ids, len(uniques))

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=22)

train_y_ohe =  one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

# 模型构建
model = Sequential([
    # 隐藏层
    Dense(10, activation='relu', input_shape=(4,)),
    # 隐藏层
    Dense(10, activation='relu'),
    # 输出层
    Dense(3, activation='softmax')]
)

# print(model.summary())
# utils.plot_model(model, show_shapes=True, to_file='model.png')

# 设置模型的相关参数：优化器，损失函数和评价指标
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

test_x = np.array(train_x, dtype=np.float32)
train_x = np.array(train_x, dtype=np.float32)

model.fit(train_x, train_y_ohe, epochs=10, batch_size=1, verbose=1)

loss, acc=model.evaluate(test_x, test_y_ohe,verbose=1)


print('loss: %s, acc: %s' % (loss, acc))