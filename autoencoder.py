from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt

def loadImages():
	imgs = []
	rootdir = 'D:/resize_FP_Dataset'
	lists = os.listdir(rootdir)
	for item in lists:
		path = os.path.join(rootdir,item)
		if(os.path.isfile(path)):
			img = image.load_img(path, target_size=(200, 200))
			x = image.img_to_array(img)
			imgs.append(x.flatten())
	return np.array(imgs) / 255

x_train = loadImages()

tf.compat.v1.reset_default_graph() # clean graph

with tf.compat.v1.name_scope('input'):
	x = tf.compat.v1.placeholder(shape = (None, x_train.shape[1]), name = 'x', dtype=tf.float32)
	y = tf.compat.v1.placeholder(shape = (None, x_train.shape[1]), name = 'y', dtype=tf.float32)

with tf.compat.v1.variable_scope('encoder1'):
	w1 = tf.compat.v1.get_variable('weight1', shape= [x_train.shape[1], 200], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
	b1 = tf.compat.v1.get_variable('bias1', shape= [200], dtype=tf.float32, initializer=tf.constant_initializer(0.0))  
	x_h1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

with tf.compat.v1.variable_scope('encoder2'):
	w2 = tf.compat.v1.get_variable('weight2', shape= [200, 50], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
	b2 = tf.compat.v1.get_variable('bias2', shape= [50], dtype=tf.float32, initializer=tf.constant_initializer(0.0))  
	x_h2 = tf.nn.relu(tf.add(tf.matmul(x_h1, w2), b2))

with tf.compat.v1.variable_scope('encoder3'):
	w3 = tf.compat.v1.get_variable('weight3', shape= [50, 10], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
	b3 = tf.compat.v1.get_variable('bias3', shape= [10], dtype=tf.float32, initializer=tf.constant_initializer(0.0))  
	x_h3 = tf.nn.relu(tf.add(tf.matmul(x_h2, w3), b3))


with tf.compat.v1.variable_scope('dencoder1'):
	w4 = tf.compat.v1.get_variable('weight4', shape= [10, 50], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
	b4 = tf.compat.v1.get_variable('bias4', shape = [50], dtype=tf.float32, initializer=tf.constant_initializer(0.0))  
	x_h4 = tf.nn.relu(tf.add(tf.matmul(x_h3, w4), b4))

with tf.compat.v1.variable_scope('dencoder2'):
	w5 = tf.compat.v1.get_variable('weight5', shape = [50, 100], dtype=tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1))
	b5 = tf.compat.v1.get_variable('bias5', shape = [100], dtype=tf.float32, initializer = tf.constant_initializer(0.0))  
	x_h5 = tf.nn.relu(tf.add(tf.matmul(x_h4, w5), b5))

with tf.compat.v1.variable_scope('dencoder3'):
	w6 = tf.compat.v1.get_variable('weight6', shape = [100, 200], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
	b6 = tf.compat.v1.get_variable('bias6', shape = [200], dtype = tf.float32, initializer = tf.constant_initializer(0.0))  
	x_h6 = tf.add(tf.matmul(x_h5, w6), b6)

with tf.compat.v1.variable_scope('output'):
	w7 = tf.compat.v1.get_variable('weight7', shape = [200, x_train.shape[1]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
	b7 = tf.compat.v1.get_variable('bias7', shape = [x_train.shape[1]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))  
	output = tf.add(tf.matmul(x_h6, w7), b7)
	print(output)


with tf.name_scope('cross_entropy'):
	loss = tf.reduce_mean(tf.pow(x - output, 2))

with tf.name_scope('train'):
	train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# print(tf.compat.v1.global_variables())

batch_size = 4
epochs = 5
tr_loss = list() #存training過程中的loss值

with tf.compat.v1.Session() as sess:
	sess.run(tf.compat.v1.global_variables_initializer())
	training_loss = 0
	for step in range(epochs):
		iterations = int(np.floor(len(x_train) / batch_size))		
		for j in np.arange(iterations):
			batch_idx_start = j * batch_size
			batch_idx_stop = (j+1) * batch_size

			x_batch = x_train[batch_idx_start : batch_idx_stop]
			y_batch = x_train[batch_idx_start : batch_idx_stop]
			_, c = sess.run([train_step, loss], feed_dict={x: x_batch})

			training_loss += c

		training_loss /= epochs
		tr_loss.append(training_loss)

		print("Epoch:", step, "cost=", "{:.9f}".format(c))

	print('--- training done ---')

	plt.plot(range(len(tr_loss)), tr_loss, label='training')
	plt.title('Loss')
	plt.legend(loc='best')

	encode_decode = sess.run(output, feed_dict={x: x_train[10:20]})
	f, a = plt.subplots(2, 10, figsize=(10, 2))
	for i in range(10):
		a[0][i].imshow(np.reshape(x_train[i], (200, 200, 3)))
		a[1][i].imshow(np.reshape(encode_decode[i], (200, 200, 3)))

	plt.show()



