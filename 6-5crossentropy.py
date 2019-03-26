import tensorflow as tf
labels = [[0,0,1],[0,1,0]]
logits = [[2,0.5,6],[0.1,0,3]]
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

