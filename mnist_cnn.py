
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as pyplot
data_path = "data/"
mnist = input_data.read_data_sets(data_path,one_hot=True)
images = mnist.train.images

trainning_rate=0.001
epoch=200
batch_size=50
display_step = 10
#num = 100

#image = images[num].reshape(28,28)
#pyplot.imshow(image)
#pyplot.show()

inputx= tf.placeholder(tf.float32,[None,28*28])
labels = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(inputx,[-1,28,28,1])
#卷积
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def maxpool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#第一层卷积 28*28*1
w_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
#第一层池化 14*14*32
pool1 = maxpool(h_conv1)

#第二层卷积 14*14*32
w_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2 = tf.nn.relu(conv2d(pool1,w_conv2) + b_conv2)
#第二层池化 7*7*64
pool2 = maxpool(h_conv2)

#展平处理
pool2_flat = tf.reshape(pool2,[-1,7*7*64])
#全连接层 1024
w_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64,1024],stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(pool2_flat,w_fc1) + b_fc1)

#全连接层10
w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024,10],stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
y_pred = tf.nn.softmax(tf.matmul(h_fc1,w_fc2) + b_fc2)

#loss
loss = -tf.reduce_sum(labels * tf.log(y_pred))
train_step = tf.train.AdamOptimizer(trainning_rate).minimize(loss)

pre = tf.equal(tf.argmax(y_pred,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(pre,tf.float32))



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={inputx:batch[0],labels:batch[1]})
        if i%display_step == 0:
            c = accuracy.eval(feed_dict={inputx:batch[0],labels:batch[1]})
            print("step:" ,i,"\taccuracy:",c)