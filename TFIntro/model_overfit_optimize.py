#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 20:28
# @Author  : ywendeng
# @Description :  使用L2范式正则化防止模型过拟合（在神经网络模型中
# 当网络结构复杂之后定义的网络结构的部分和计算损失函数那部分可能不在同一个函数中
# 这样通过变量的方式计算损失函数就不太方便了-----使用TensorFlow 中的集合Collection来解决这个问题）

'''
  全连接神经网络的定义：在损失函数中加入正则化防止模型过拟合
'''
import tensorflow as tf
from numpy.random import RandomState


# 获取一层神经网络的权重，并将其L2正则化加入"losses"的损失函数中
def get_weight(shape, lamda):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将正则化损失项加入集合
    tf.add_to_collection("losses",
                         tf.contrib.layers.l2_regularizer(lamda)(var))
    return var


# 设置batch,确定每次选择多少训练数据作为模型输入
batch_size = 8
# 设置训练数据集的大小
dataset_size = 128
# 设置模型训练次数
train_steps = 100000
# 定义网络结构
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

# 设置网络的层级
layer_dimension = [2, 10, 10, 10, 1]
cur_layer = x
in_dimension = layer_dimension[0]
# 循环构造网络结构
for i in range(1, len(layer_dimension)):
    out_dimension = layer_dimension[i]
    weights = get_weight([in_dimension, out_dimension], 0.001)
    # 设置偏值项
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用使用激活函数函数来更新当前层的输出
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights) + bias)
    in_dimension = out_dimension
# 计算损失函数
loss_mse = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误误差加入到集合中
tf.add_to_collection("losses", loss_mse)
# 从集合中取出losses 集合列表，将列表中的各个元素加起来，就为损失函数
loss = tf.add_n(tf.get_collection("losses"))
# 设置学习率为指数衰减
# learning_rate = tf.train.exponential_decay(0.1, train_steps,
#                                           dataset_size / batch_size, 0.96, staircase=True)
# 设置反向优化算法
train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

# 初始化输入训练数据集
rdm = RandomState(1)
X = rdm.rand(dataset_size, 2)
# 为样本数据集设置标签如果x1+x2<1则为正样本，否则为负样本，其中正样本使用1表示，负样本使用0表示
Y = [[int(x1 + x2 < 1)] for x1, x2 in X]

# 构建会话来
with tf.Session() as sess:
    init_ops = tf.global_variables_initializer()
    sess.run(init_ops)

    # 开始进行模型训练
    for i in range(train_steps):
        # 每次从训练数据中选择batch个子数据作为模型训练
        start = (i * batch_size) % dataset_size
        # 主要是为了防止下标越界
        end = min(start + batch_size, dataset_size)
        # 通过选择样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
            print("after %d training steps, cross entropy on all data is %g" %
                  (i, total_cross_entropy))
