#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/21 14:32
# @Author  : ywendeng
# @Description :
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
# a、b 分别为两个张量，此处对张量的使用，便于加快计算速度，同时，增加程序的可读性
result = a + b

with tf.Session() as sess:
    # 将该回话设置为默认的回话
    # with 语句适用于对资源进行访问的场合，
    # 确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，
    # 比如文件使用后自动关闭、线程中锁的自动获取和释放等。
    with sess.as_default():
        print(result.eval())
