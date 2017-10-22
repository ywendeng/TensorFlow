#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/22 14:39
# @Author  : ywendeng
# @Description : 使用TensorFlow来训练神经网络模型参数

'''
 训练模型步骤如下：
 1. 定义神经网络模型结构和前向传播的输出结果
 2. 定义损失函数以及选择反向传播优化的算法
 3. 生成回话并在训练数据上反复运行反向传播优化算法
'''
import tensorflow as tf
from numpy.random import RandomState

# 设置batch,确定每次选择多少训练数据作为模型输入
batch_size = 8
# 设置训练数据集的大小
dataset_size = 128
# 设置模型训练次数
train_steps = 50000

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="x_input")
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y_output")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 构建损失函数和反向优化算法
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
)
train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)

# 初始化输入训练数据集
rdm = RandomState(1)
X = rdm.rand(dataset_size, 2)
# 为样本数据集设置标签如果x1+x2<1则为正样本，否则为负样本，其中正样本使用1表示，负样本使用0表示
Y = [[int(x1+x2<1)] for x1, x2 in X]

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
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("after %d training steps, cross entropy on all data is %g" %
                  (i, total_cross_entropy))

    # 输出训练参数结果
    print(sess.run(w1))
    print("==========================")
    print(sess.run(w2))
