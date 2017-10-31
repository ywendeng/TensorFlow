#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/27 16:35
# @Author  : ywendeng
# @Description :
from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf

def load_data():
    mnist = input_data.read_data_sets("DataSet/MNIST/", one_hot=True)
    # 打印Train data size 55000
    print("Train data size:",mnist.train.num_examples)
    # 打印validating 数据5000
    print("Validating data size: ",mnist.validation.num_examples)
    # 打印Testing data size: 1000
    print("Validating data size: ", mnist.test.num_examples)
    # 打印Example training data。 每张图片被映射成1*784的一维数组
    print("Example training data :",mnist.train.images[0])
    # 打印Example training data label
    print("Example training data label: ",mnist.train.labels[0])


if __name__ == '__main__':
    load_data()
