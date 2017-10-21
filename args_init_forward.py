#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/21 20:06
# @Author  : ywendeng
# @Description : 全连接神经网络中的参数初始化和前向传播算法
# 构造计算图
import tensorflow as tf

# 设置权重值
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 设置输入特征
x = tf.constant([[0.5, 0.9]])

# 实现各节点和矩阵运算------注意在tensorFlow中调用matmul，则其纬度必须相同
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 调用实现初始化所有的权重
all_init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(all_init)
    with sess.as_default():
        print(y.eval())
