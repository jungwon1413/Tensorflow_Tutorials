import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32) # initial weight: 0.3
b = tf.Variable([-.3])   # initial bias: -0.3, implicitly float32 type

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)


# loss
loss = tf.reduce_sum(tf.square(linear_model - y))   # tf.sum이 아니라 tf.reduce_sum인 이유가 있다.
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)    # Optimizer: Gradient Descent (수치: 0.01)
train = optimizer.minimize(loss)    # loss를 minimize하는것이 train이라고 정의내림


# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training_loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)   # reset values to wrong
for i in range(1000):    # train이 1000번 반복된다.
	sess.run(train, {x: x_train, y: y_train})    # train에 x 전부와 y 전부를 넣어준다.


# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s, b: %s, loss: %s" % (curr_W, curr_b, curr_loss))



# NumPy is often used to load, manupulate and preprocess data.
import numpy as np

# Declare list of features. We only have one numeric feature.
# There are many other types of columns that are more
# complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape[1])]


# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)


# Tensorflow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how bit each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# Returns input function that would feed dict of numpy arrays into the model.
input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)


# We can invoke 1000 training steps by invoking the method and
# passing the training data set.
estimator.train(input_fn=input_fn, steps=1000)


# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)