# -*- coding: utf-8 -*-
# @Time : 2023/5/10 1:19
# @Author : ZuoWenBin


# 绘图
import seaborn as sns
# 数值计算
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# sklearn中的相关⼯具
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
# 逻辑回归
from sklearn.linear_model import LogisticRegressionCV
# tf.keras中使⽤的相关⼯具
# ⽤于模型搭建
from tensorflow.keras.models import Sequential
# 构建模型的层和激活⽅法
from tensorflow.keras.layers import Dense, Activation
# 数据处理的辅助⼯具
from tensorflow.keras import utils

# 1.获取数据集
# iris = sns.load_dataset("iris")
file = "iris_dataset/iris.csv"
iris = pd.read_csv(file)

iris = iris.drop("Unnamed: 0", axis = 1)


# print(iris.head())
# print(type(iris))
# # 展示前五行
# print(iris.head())

# 将数据间的关系可视化
sns.pairplot(iris, hue = 'Species')
plt.show()

# 2.数据预处理
# 花瓣和花萼的数据
x = iris.values[:, :4]
# 标签值
y = iris.values[:, 4]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# 3.特征工程
# 4.模型构建
# 实例化分类器
lr = LogisticRegressionCV()
# 训练
lr.fit(train_x, train_y)
# 5.模型训练
# 6.模型评估
# 计算准确率并进⾏打印
# print("Accuracy = {:.2f}".format(lr.score(test_x, test_y)))
