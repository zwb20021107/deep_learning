# -*- coding: utf-8 -*-
# @Time : 2023-08-14 12:57
# @Author : ZuoWenBin


import tensorflow as tf
import numpy as np

# 定义两个二维张量 a和 b
a = tf.constant([[1, 2],
                 [4, 5]])
b = tf.constant([[1, 1],
                 [1, 1]])



# 数学运算
print(tf.add(a, b))
print(tf.multiply(a, b)) # 元素相乘
print(tf.matmul(a, b)) # 矩阵相乘

# 聚合运算
# 求和
print(tf.reduce_sum(a, axis=0))
# 求平均值
print(tf.reduce_mean(a))
# 求最大值
print(tf.reduce_max(a))

# 求最小值
print(tf.reduce_min(a))

# 求最大值索引
print(tf.argmax(a))

#  求最小值索引
print(tf.argmin(a))

