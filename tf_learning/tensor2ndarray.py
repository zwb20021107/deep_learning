# -*- coding: utf-8 -*-
# @Time : 2023-08-14 12:53
# @Author : ZuoWenBin

import tensorflow as tf
import numpy as np

# 将张量转换为numpy数组
rank_2_tensor = tf.constant([[1, 2], [3, 4]])

# 方法一
ndarray1 = np.array(rank_2_tensor)
print(ndarray1)

# 方法二
ndarray2 = rank_2_tensor.numpy()
print(ndarray2)