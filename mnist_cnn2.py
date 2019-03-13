#!/usr/bin/env python
# encoding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
FLAGS = None

def main():
    data_path = "data/"
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1,shape = shape)
        return tf.Variable(initial)

    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

