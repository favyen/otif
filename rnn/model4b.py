import numpy
import tensorflow as tf
#tf.disable_eager_execution()
import os
import os.path
import random
import math
import time

BATCH_SIZE = 32
MAX_LENGTH = 128
NUM_BOXES = 16

class Model:
	def _fc_layer(self, name, input_var, input_size, output_size, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		batchnorm = options.get('batchnorm', False)

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights',
				shape=[input_size, output_size],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / input_size)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[output_size],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			output = tf.matmul(input_var, weights) + biases
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.float32, [None, MAX_LENGTH, 10])
		self.boxes = tf.placeholder(tf.float32, [None, MAX_LENGTH, NUM_BOXES, 5])
		self.mask = tf.placeholder(tf.float32, [None, MAX_LENGTH])
		self.targets = tf.placeholder(tf.float32, [None, MAX_LENGTH, NUM_BOXES])
		self.learning_rate = tf.placeholder(tf.float32)

		batch_size = tf.shape(self.inputs)[0]

		def rnn_step(prev_state, cur_input):
			with tf.variable_scope('rnn_step', reuse=tf.AUTO_REUSE):
				pairs = tf.concat([prev_state, cur_input], axis=1)
				rnn1 = self._fc_layer('rnn1', pairs, 64+10, 128)
				rnn2 = self._fc_layer('rnn2', rnn1, 128, 128)
				rnn3 = self._fc_layer('rnn3', rnn2, 128, 128)
				rnn4 = self._fc_layer('rnn4', rnn3, 128, 128, {'activation': 'none'})
				return rnn4[:, 0:64], rnn4[:, 64:128]

		cur_state = tf.zeros([batch_size, 64], dtype=tf.float32)
		rnn_outputs = []
		for i in range(MAX_LENGTH):
			cur_state, cur_outputs = rnn_step(cur_state, self.inputs[:, i, :])
			rnn_outputs.append(cur_outputs)
		rnn_outputs = tf.stack(rnn_outputs, axis=1)

		features = tf.concat([rnn_outputs, self.inputs], axis=2)
		features = tf.reshape(features, [batch_size, MAX_LENGTH, 1, 64+10])
		features_tiled = tf.tile(features, [1, 1, NUM_BOXES, 1])
		flat_features_tiled = tf.reshape(features_tiled, [batch_size*MAX_LENGTH*NUM_BOXES, 64+10])
		flat_boxes = tf.reshape(self.boxes, [batch_size*MAX_LENGTH*NUM_BOXES, 5])
		fc_cat = tf.concat([flat_features_tiled, flat_boxes], axis=1)
		fc1 = self._fc_layer('fc1', fc_cat, 64+10+5, 64)
		fc2 = self._fc_layer('fc2', fc1, 64, 64)
		fc3 = self._fc_layer('fc3', fc2, 64, 64)
		fc4 = self._fc_layer('fc4', fc3, 64, 1, {'activation': 'none'})
		self.pre_outputs = tf.reshape(fc4, [batch_size, MAX_LENGTH, NUM_BOXES])
		self.pre_outputs = tf.concat([
			-6*tf.ones([batch_size, MAX_LENGTH, 1], dtype=tf.float32),
			self.pre_outputs[:, :, 1:],
		], axis=2)
		self.outputs = tf.nn.softmax(self.pre_outputs)
		self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.pre_outputs) * self.mask
		self.loss = tf.reduce_sum(self.loss, axis=1) / tf.reduce_sum(self.mask, axis=1)
		self.loss = tf.reduce_mean(self.loss)

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
