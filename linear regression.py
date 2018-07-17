import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# y = 2x^2 + 3x +c

tf.reset_default_graph()
x = np.linspace(0, 1, 100)
y_true = 2*x*x + 3*x + np.random.rand(len(x))


w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype = tf.float32)
w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype = tf.float32)
b1 = tf.Variable(tf.zeros([1]))

y_pred = w1*x*x + w2*x + b1

loss = tf.reduce_mean(tf.square(y_pred - y_true))

optim = tf.train.GradientDescentOptimizer(0.1)

train_optim = optim.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in np.arange(500):
    sess.run(train_optim)
    if step % 20 == 0:
        print("step: " + str(step) + ", W1: " + str(sess.run(w1)[0]) + ", W2: " + str(sess.run(w2)[0]) + ", b1: " + str(sess.run(b1)[0]))
y_out = sess.run(y_pred)
plt.plot(y_true, y_out, 'g.')
plt.show()

plt.plot(x, y_true, 'b.')
plt.plot(x, y_out, 'r.')
plt.show()
sess.close()
