# -*- coding: utf-8 -*-
# @Time : 2023-08-14 13:07
# @Author : ZuoWenBin

import tensorflow as tf

# 创建一维变量
my_variable = tf.Variable([[1, 2],
                           [3, 4]])

print(my_variable)



# 常见常量
print(my_variable.shape)
print(my_variable.dtype)
print(my_variable.name)