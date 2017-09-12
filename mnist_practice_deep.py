from tensorflow.examples.tutorials.mnist import input_data

# mnist is a lightweight class which stores
# the training, validation, and testing sets as NumPy arrays.
# It also provides a function for iterating through data minibatches.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Why use session: tensorflow relies on C++ backend. session is a connection to this backend.
# Common usage: create graph and launch it to a session. 
# (Make an outline with Python -> send it to C++ backend)
import tensorflow as tf
sess = tf.InteractiveSession()


# building the computatino graph by creating nodes (for the images)
# shape argument is optional, but good for catching bugs from inconsistent tensor shapes.
x = tf.placeholder(tf.float32, shape=[None, 784])    # 2D Tensor, None: batch size!
y_ = tf.placeholder(tf.float32, shape=[None, 10])    # 2D Tensor

# Variable setting. It can be used and even modified by the computation.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Before you use variables, you must initialize them first, using session. (This case, global.)
sess.run(tf.global_variables_initializer())

# Definition of our model. (In our case, linear regression model.)
y = tf.matmul(x, W) + b

# Specify a loss function
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Training step. (Check out variety of built-in optimization algorithms!)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # learning rate: 0.5?


# Iteration of training
for _ in range(1000):
	batch = mnist.train.next_batch(100)    # Stochastic
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})  # pixel of image, classification of prediction


# Evaluation process: how well did our model do?
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # about 92%



## Build a Multilayer Convolutional Network ##
