from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf


x = tf.placeholder(tf.float32, [None, 784]) # total pixel: ? * ? = 784
W = tf.Variable(tf.zeros([784, 10]))  # total 784 pixel, 10 different classification
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):   # 1000 Epoch
	batch_xs, batch_ys = mnist.train.next_batch(100)    # 100 random data per batch (stochastic training)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# (y: actual answer, y_(read as y-hat): prediction)
# boolean of whether prediction is correct or not.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# [True, False, True, True] → [1, 0, 1, 1] = 0.75 (mean)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))