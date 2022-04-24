import numpy
import tensorflow as tf
#tf.disable_eager_execution()
import os
import os.path
import random
import math
import time

class Model:
	def __init__(self, width, height):
		self.inputs = tf.placeholder(tf.uint8, [None, None, None, 3])
		self.targets = tf.placeholder(tf.float32, [None])
		self.learning_rate = tf.placeholder(tf.float32)
		self.is_training = tf.placeholder(tf.bool)

		cur = tf.image.resize(tf.cast(self.inputs, tf.float32)/255.0, [height, width])
		for features in [32, 64, 64, 64, 64, 64, 64]:
			cur = tf.keras.layers.Conv2D(
				features, (4, 4),
				strides=2, activation='relu', padding='same'
			)(cur)
		self.outputs = tf.keras.layers.Conv2D(
			1, (4, 4),
			padding='same'
		)(cur)[:, 0, 0, 0]
		self.loss = tf.reduce_mean(tf.square(self.outputs - self.targets))

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep=None)
