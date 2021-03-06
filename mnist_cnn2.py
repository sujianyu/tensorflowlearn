#!/usr/bin/env python
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import sys
FLAGS = None

def main(_):
    data_path = "data/"
    print("FLAGS.data_dir",FLAGS.data_dir)
    ckpt_path = FLAGS.ckpt_dir
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



    x_image = tf.reshape(x,[-1,28,28,1])

    wconv1 = weight_variable([5,5,1,32])
    bconv1 = bias_variable([32])

    hconv1 = tf.nn.relu(conv2d(x_image,wconv1) + bconv1)
    pool1 = max_pool_2x2(hconv1)

    wconv2 = weight_variable([5,5,32,64])
    bconv2 = bias_variable([64])
    hconv2 = tf.nn.relu(conv2d(pool1,wconv2) + bconv2)
    pool2 = max_pool_2x2(hconv2)

    #构造全连接网络
    wfc1 = weight_variable([7*7*64,1024])
    bfc1 = bias_variable([1024])
    hpool2_flat = tf.reshape(pool2,[-1,7*7*64])
    hfc1 = tf.nn.relu(tf.matmul(hpool2_flat,wfc1) + bfc1)

    #Dropout
    keep_prob = tf.placeholder(tf.float32)
    hfc1_drop = tf.nn.dropout(hfc1,keep_prob)

    #构造全连接 10
    wfc2 = weight_variable([1024,10])
    bfc2 = bias_variable([10])
    y_conv = tf.matmul(hfc1_drop,wfc2) + bfc2

    global_step = tf.Variable(0,name="global_step",trainable=False)

    #损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_conv))

    #优化函数
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)

    #准确率
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    saver = tf.train.Saver(max_to_keep=1)
    saver_max_acc = 0
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
           saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))
           print("Restore from CKPT")
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            _,step = sess.run([train_step,global_step], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            i=step
            if i%100 ==0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})
                print("step %d,trainning accuracy %g,i=%d" % (step,train_accuracy,i))
                #保存模型
                saver.save(sess, ckpt_path + "mnist.ckpt", global_step=i+1)




        model_file = tf.train.latest_checkpoint(ckpt_path)
        saver.restore(sess,model_file)

        print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="data/",help="Direcotry for mnist data.")
    parser.add_argument("--ckpt_dir",type=str,default="ckpt/",help = "Directory ro saver.")
    FLAGS,unparsed = parser.parse_known_args(sys.argv[1:])

    tf.app.run()