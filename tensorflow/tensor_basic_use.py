# -*- coding: utf-8 -*-
# @Time : 2023/5/10 0:46
# @Author : ZuoWenBin

import tensorflow as tf


# tf.constant(value,dtype,shape)
# value用来指定数据，dtype用来显式地声明数据类型，shape用来指定数据的形状，

# 创建int32类型的0维张量
rank_0_tensor = tf.constant(4)
# print(rank_0_tensor)
# 创建float类型的1维张量
rank_1_tensor = tf.constant([1.0, 2.0])
# print(rank_1_tensor)
# 创建float16类型的⼆维张量
rank_2_tensor = tf.constant([[1, 2],[3, 4],[5, 6]], dtype= tf.float16)
# print(rank_2_tensor)


# 转换维numpy形式
# np.array(tensor)
# tensor.numpy()

# 法1：
import numpy as np
# print(np.array(rank_2_tensor))

# 法2：
# print(rank_2_tensor.numpy())


a = tf.constant([[1, 2], [2, 3]])
b = tf.constant([[2, 1], [3, 2]])

print(a + b)
print(tf.add(a, b))
print(a - b)
print(tf.subtract(a, b))
print(a * b)
print(tf.multiply(a, b))

# 矩阵乘法
print(tf.matmul(a, b))


# tf.reduce_sum() # 求和
# tf.reduce_mean() # 平均值
# tf.reduce_max() # 最⼤值
# tf.reduce_min() # 最⼩值
# tf.argmax() # 最⼤值的索引
# tf.argmin() # 最⼩值的索引

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# 最大值
print(tf.reduce_max(c))
# 最大值索引
print(tf.argmax(c))

# 最小值
print(tf.reduce_min(c))
# 最值索引
print(tf.argmin(c))

# 求和
print(tf.reduce_sum(c))
# 求平均值
print(tf.reduce_mean(c))

# tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称

a = tf.Variable([[1, 2], [3, 4]], name = 'b')

print("Shape: ",a.shape)
print("DType: ",a.dtype)
print("As NumPy: ", a.numpy)