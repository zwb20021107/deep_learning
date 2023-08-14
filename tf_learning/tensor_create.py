# -*- coding: utf-8 -*-
# @Time : 2023-08-14 12:37
# @Author : ZuoWenBin


import tensorflow as tf
import numpy as np

# 创建一个int32型的零维张量 dtype设置数据类型
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# 创建一个int32型的一维张
rank_1_tensor = tf.constant([1, 2])
print(rank_1_tensor)

# 创建一个int32型的二维张量
rank_2_tensor = tf.constant([[1, 2], [3, 4]])
print(rank_2_tensor)

# 创建一个int32型的三维张量
rank_3_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int64)
print(rank_3_tensor)