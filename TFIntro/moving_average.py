#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/25 11:44
# @Author  : ywendeng
# @Description : 在采用随机梯度下降算法训练神经网络模型时，使用滑动平均模型在很多运用中都可以
# 在一定程度提高最终模型在测试数据上的表现
# 每次更新时影子变量的值会更新未：shadow_variable= decay*shadow_variable+(1-decay)*variable
import tensorflow as tf

# 定义一个变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)
# 这里step变量模拟神经网络中迭代的轮数，可以用于控制衰减率
step = tf.Variable(0, trainable=False)
# 定义一个滑动平均的类，衰减率初始化为0.99，step为迭代次数
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时，这个列表中的变量
# 都会被更新
maintain_average_op = ema.apply([v1])

# 创建回话
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 获取变量初始化v1的值和滑动平均之后的取值
    print(sess.run([v1, ema.average(v1)]))
    # 更新v1的值为5
    sess.run(tf.assign(v1,5))
    # 更新v1的滑动平均值,衰减率为min{0.99,(1+step)/(10+step)=0.1}=0.1
    # 所以v1的滑动平均值会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

