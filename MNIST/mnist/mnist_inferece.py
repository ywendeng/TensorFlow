#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/30 20:24
# @Author  : ywendeng
# @Description : 主要使用tesorflow 参数管理定义了神经网络的结构参数和前向传播过程

import tensorflow as tf

# 设置mnist的相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数，在本例中使用只有一个隐藏层的网络结构


# 使用get_variable 定义参数
def get_weight(shape, regularizer):
    '''
    在训练神经网络时创建遍量; 在测试时通过保存的模型加载这些变量的取值,
    而且更加方便的是,因为可以在变量加载时滑动平均变量重命名，所以可以直接
    通过同样的名字在训练时使用变量自身，而在测试时使用变量的滑动平均值。这个函数
    也同时将变量的正则化损失函数加入损失集合
    '''
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 给出了正则化损失函数之后，将当前变量的正则化损失加入损失结合
    if regularizer != None:
        tf.add_to_collection("losess", regularizer(weights))
    return weights


# 定义了前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络以及前向传播过程
    with tf.variable_scope("layer1"):
        # 获取对应的权重
        weights = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
        # 生成偏值项
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)) + biases
    # 声明第二层神经网络以及前向传播过程
    with tf.variable_scope("layer2"):
        weights = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
        # 生成偏值项
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
