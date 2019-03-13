# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

datapath = "data/"

mnist = input_data.read_data_sets(datapath,one_hot=True)

print("read minist data_set end")
learning_rate = 0.001
trainning_epoch = 25
batch_size = 100
display_step = 1

n_hidden1 = 256
n_hidden2 = 256
n_hidden3 = 256

n_input = 784
n_classes = 10

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

def multilayer_perception(x,weights,biases):
    layer1 = tf.add(tf.matmul(x,weights["h1"]),biases["h1"])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1,weights["h2"]),biases["h2"])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2,weights["h2"]),biases["h2"])
    layer3 = tf.nn.relu(layer3)

    outlayer = tf.add(tf.matmul(layer2,weights["out"]),biases["out"])
    return outlayer

weights = {
    "h1": tf.Variable(tf.random_normal([n_input,n_hidden1])),
    "h2": tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    "out":tf.Variable(tf.random_normal([n_hidden2,n_classes]))
}

biases = {
    "h1" : tf.Variable(tf.random_normal([n_hidden1])),
    "h2": tf.Variable(tf.random_normal([n_hidden2])),
    "out": tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perception(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(trainning_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c/total_batch
        if epoch % display_step ==0 :
            print("Epoch:",'%04d' % (epoch +1),"{:.9f}".format(avg_cost))
    print("Finished!")

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))