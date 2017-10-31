#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/30 21:29
# @Author  : ywendeng
# @Description : 神经网络的训练过程

import tensorflow as tf
from   MNIST.mnist import mnist_inferece
from tensorflow.examples.tutorials.mnist import input_data
import os

# 配置神经网络的参数
BATCH_SIZE = 100  # 一个训练batch中训练数据个数，数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习利率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
# 设置模型保存的名称和路径
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


# 定义模型结构
def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inferece.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inferece.OUTPUT_NODE], name='y_input')
    # 声明L2正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None
    y = mnist_inferece.inference(x, regularizer)
    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率和训练路数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所代表神经网络参数的变量上使用滑动平均。其中在tf.trainable_variables 返回的就是没有指定
    # trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵作为刻画预测值和真实值之间的差距函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    # 计算在当前batch中所有样例交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算正则化的损失,一般只计算神经网络边上的权重的正则化损失，而不适用偏置项

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losess"))
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                               , global_step
                                               , mnist.train.num_examples / BATCH_SIZE
                                               , LEARNING_RATE_DECAY)
    # 定义优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 在训练神经网络模型时，每过一遍数据即需要通过反向传播来更新神经网络中的参数，又要更新每个参数的滑动平均值
    train_op = tf.group(train_step, variable_averages_op)
    # 初始化持久化类
    saver = tf.train.Saver()
    # 初始化回话并进行模型训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 产生这一轮使用的一个batch训练数据，并运行训练过程

        # 在模型训练过程中，不在测试模型在验证数据上的表现，验证和测试将在独立的程序中完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                # 输出当前的训练情况，只输出模型在当前batch 损失函数的大小
                print("After %d training steps ,"
                      "loss on training batch is %f" % (i, loss_value))
                # 保存当前模型，此处给出了global_step 参数，这样可以让每个被保存的模型的文件名末尾加上
                # 训练的轮数 比如"model.ckpt-1000" 表示训练1000次之后得到模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# 主程序入口
def main(argv=None):
    # 声明处理Mnist数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("DataSet", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    # 在TensorFlow中定义该方法，会主动调用main方法
    tf.app.run()
