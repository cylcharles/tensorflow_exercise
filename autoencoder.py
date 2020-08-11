from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv2DTranspose
import cv2

def loadImages():
	x_imgs = []
	y_imgs = []
	rootdir = 'path'
	lists = os.listdir(rootdir)
	for item in lists:
		if os.path.splitext(item)[1] == '.jpg':
			path = os.path.join(rootdir,item)
			if(os.path.isfile(path)):
				img = image.load_img(path, target_size=(512, 512))
				x = image.img_to_array(img)
				x_imgs.append(x)
				ret,thresh = cv2.threshold(x, 127, 255, cv2.THRESH_TRUNC)
				y_imgs.append(thresh)
				# imgs.append(x.flatten())
	return np.array(x_imgs) / 255, np.array(y_imgs) / 255

x_train, y_train = loadImages()

print("x_train = ", x_train.shape)
print("y_train = ", y_train.shape)

tf.compat.v1.reset_default_graph() # clean graph

with tf.compat.v1.name_scope('input'):
	x = tf.compat.v1.placeholder(shape = (None, 512, 512, 3), name = 'x', dtype=tf.float32)
	y = tf.compat.v1.placeholder(shape = (None, 512, 512, 3), name = 'y', dtype=tf.float32)

with tf.compat.v1.variable_scope('encoder'):
	conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')

	conv3 = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')

	conv5 = tf.layers.conv2d(inputs=maxpool4, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	conv6 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	encoded = tf.layers.max_pooling2d(conv6, pool_size=(2,2), strides=(2,2), padding='same')

with tf.compat.v1.variable_scope('dencoder'):

	upsample1 = tf.image.resize_images(encoded, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv5 = tf.layers.conv2d(inputs=upsample1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

	upsample2 = tf.image.resize_images(conv5, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv6 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

	upsample3 = tf.image.resize_images(conv6, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv7 = tf.layers.conv2d(inputs=upsample3, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

	upsample4 = tf.image.resize_images(conv7, size=(256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv8 = tf.layers.conv2d(inputs=upsample4, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

	upsample5 = tf.image.resize_images(conv8, size=(512,512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv9 = tf.layers.conv2d(inputs=upsample5, filters=3, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

	decoded  = tf.layers.conv2d(inputs=conv9, filters=3, kernel_size=(3,3), padding='same', activation=None)

# with tf.compat.v1.variable_scope('dencoder2'):
# 	# w6 = tf.compat.v1.get_variable('weight6', shape = [100, 200], dtype=tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1))
# 	# b6 = tf.compat.v1.get_variable('bias6', shape = [200], dtype=tf.float32, initializer = tf.constant_initializer(0.0))
# 	# x_h6 = tf.nn.relu(tf.add(tf.matmul(x_h5, w6), b6))
# 	w6 = tf.compat.v1.get_variable('weight5', shape= [64, 5, 5, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
# 	b6 = tf.compat.v1.get_variable('bias5', shape = [128], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
# 	x_h6 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(x_h5, w6, [1, 128, 128, 128], strides=[1, 2, 2, 1], padding='SAME'), b6))


with tf.name_scope('cross_entropy'):
	cost = tf.reduce_mean(tf.pow(y - decoded, 2))
	# cost = tf.losses.mean_squared_error(y, decoded)
	# cost = tf.losses.mean_squared_error(y, decoded) - (0.5 * tf.reduce_sum(1 + encoded - tf.math.pow(mu, 2) - tf.exp(encoded)))


with tf.name_scope('train'):
	train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# print(tf.compat.v1.global_variables())

batch_size = 4
epochs = 50
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
			y_batch = y_train[batch_idx_start : batch_idx_stop]
			_, c = sess.run([train_step, cost], feed_dict={x: x_batch, y: y_batch})

			training_loss += c

		training_loss /= epochs
		tr_loss.append(training_loss)

		print("Epoch:", step, "cost=", "{:.9f}".format(c))

	print('--- training done ---')

	plt.plot(range(len(tr_loss)), tr_loss, label='training')
	plt.title('Loss')
	plt.legend(loc='best')

	encode_decode = sess.run(decoded, feed_dict={x: x_train[0:5]})
	f, a = plt.subplots(2, 5, figsize=(5, 2))
	for i in range(5):
		a[0][i].imshow(np.reshape(x_train[i], (512, 512, 3)))
		a[1][i].imshow(np.reshape(encode_decode[i], (512, 512, 3)))

	plt.show()
