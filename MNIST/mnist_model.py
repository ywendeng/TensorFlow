#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 14:06
# @Author  : ywendeng
# @Description : 使用TensorFlow 程序来解决MNIST手写体数字识别问题，再本例中主要
# 涉及以下几个问题：
# 1. 神经网络结构： 使用激活函数实现神经网络的模型的去线性化，
#    并使用一个或多个隐藏层使得神经网络的结构更深，以解决复杂问题
# 2. 模型训练： 使用带指数衰减的学习率设置、使用正则化来避免过度拟合、以及使用滑动
# 平均模型来使得最终模型更加健壮

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置mnist的相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数，在本例中使用只有一个隐藏层的网络结构
BATCH_SIZE = 100  # 一个训练batch中训练数据个数，数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习利率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 定义一个辅助函数，给定神经网络的输入和所有参数，计算神经网络前向传播结果,在这里定义一个RELU激活函数的
# 三层全连接神经网络。通过加入隐藏层实现多层网络结构，通过RELU激活函数实现去线性化，在函数中也支持传入
# 用于计算参数平均值的类，这样方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有使用滑动平均模型时，直接使用啊参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average 函数来计算得出变量的滑动平均值，然后再计算前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +
                            avg_class.average(biases1))

        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 定义模型结构
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率和训练路数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所代表神经网络参数的变量上使用滑动平均。其中在tf.trainable_variables 返回的就是没有指定
    # trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    # 计算交叉熵作为刻画预测值和真实值之间的差距函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    # 计算在当前batch中所有样例交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 声明L2正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算正则化的损失,一般只计算神经网络边上的权重的正则化损失，而不适用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                               , global_step
                                               , mnist.train.num_examples / BATCH_SIZE
                                               , LEARNING_RATE_DECAY)
    # 定义优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 在训练神经网络模型时，每过一遍数据即需要通过反向传播来更新神经网络中的参数，又要更新每个参数的滑动平均值
    train_op = tf.group(train_step, variable_averages_op)
    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个运算首先将一个布尔类型的数值转换为实数类型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化回话并进行模型训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 准备验证数据。 一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和判断训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据集上的结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps ,validation accuracy "
                      "using average model is %f" % (i, validate_acc))
                # 产生这一轮使用的一个batch训练数据，并运行训练过程
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 在训练结束之后， 在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %f " % (TRAINING_STEPS, test_acc))

# 主程序入口
def main(argv=None):
    # 声明处理Mnist数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("DataSet/MNIST/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    # 在TensorFlow中定义该方法，会主动调用main方法
    tf.app.run()
