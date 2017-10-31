#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/31 9:29
# @Author  : ywendeng
# @Description : 在模型训练中每训练1000次保存一次，这样可以通过单独的测试程序
# 更加方便地在滑动平均模型上做测试
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from MNIST.mnist import mnist_inferece
from MNIST.mnist import mnist_train

# 每隔一段时间加载一次最新模型，并在测试集上测试最新模型的正确率
EVAL_INTERVAL_SEC = 20


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inferece.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inferece.OUTPUT_NODE], name='y_input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 直接通过调用封装好的函数来计算前向传播结果，因为在测试时不关注正则化的损失值，所以此处的损失函数设置为None
        y = mnist_inferece.inference(x, None)
        # 检验使用了滑动平均模型的神经网络前向传播结果是否正确
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 这个运算首先将一个布尔类型的数值转换为实数类型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用滑动平均函数来获取平均值了
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        # 每隔EVAL_INTERVAL_SEC 秒调用一次计算正确率的过程，以检测训练过程中正取率的变化
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state 函数会通过checkpoint文件自动找到目录中的最新文件名
                ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver. restore(sess,ckpt.model_checkpoint_path)
                    # 通过文件名获取训练次数
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score= sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training steps ,validation accuracy "
                          "using average model is %f" % (global_step,accuracy_score))
                else:
                    print("no checkpoint found")
                    return
                # 每执行一次之后，线程休息一段时间
                time.sleep(EVAL_INTERVAL_SEC)
# 主程序入口
def main(argv=None):
    # 声明处理Mnist数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST/DataSet/MNIST", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    # 在TensorFlow中定义该方法，会主动调用main方法
    tf.app.run()
