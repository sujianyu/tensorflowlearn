
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as pyplot
slim = tf.contrib.slim

data_path = "data/"
mnist = input_data.read_data_sets(data_path,one_hot=True)
images = mnist.train.images

trainning_rate=0.001
epoch=200
batch_size=100
display_step = 10
#num = 100

inputx = tf.placeholder(tf.float32,[None,28*28])
labels = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(inputx,[-1,28,28,1])
def cnnsample(input):
    net = slim.conv2d(input,32,[5,5])
    net = slim.max_pool2d(net,[2,2])
    net = slim.conv2d(net,64,[5,5])
    net = slim.max_pool2d(net,[2,2])

    net = slim.flatten(net)
    net = slim.fully_connected(net,1024)
    net = slim.fully_connected(net,10,activation_fn=tf.nn.softmax)
    return net

y_pred= cnnsample(x_image)
#loss
loss = -tf.reduce_sum(labels * tf.log(y_pred))

pre = tf.equal(tf.argmax(y_pred,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(pre,tf.float32))

train_step = tf.train.AdadeltaOptimizer(trainning_rate).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={inputx:batch[0],labels:batch[1]})
        if i%display_step == 0:
            c = accuracy.eval(feed_dict={inputx:batch[0],labels:batch[1]})
            print("step:" ,i,"\taccuracy:",c)