import numpy
import tensorflow as tf
#tf.disable_eager_execution()
import os
import os.path
import random
import math
import time

BATCH_SIZE = 32
MAX_LENGTH = 64
NUM_BOXES = 8
KERNEL_SIZE = 3

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		padding = options.get('padding', 'SAME')
		batchnorm = options.get('batchnorm', False)
		transpose = options.get('transpose', False)

		with tf.variable_scope(name) as scope:
			if not transpose:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels]
			else:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, out_channels, in_channels]
			kernel = tf.get_variable(
				'weights',
				shape=filter_shape,
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			if not transpose:
				output = tf.nn.bias_add(
					tf.nn.conv2d(
						input_var,
						kernel,
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			else:
				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]
				output = tf.nn.bias_add(
					tf.nn.conv2d_transpose(
						input_var,
						kernel,
						[batch, side * stride, side * stride, out_channels],
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
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

	def reid_net(self):
		self.raw_images1 = tf.placeholder(tf.uint8, [None, 64, 64, 3])
		self.raw_images2 = tf.placeholder(tf.uint8, [None, 64, 64, 3])

		images1 = tf.cast(self.raw_images1, tf.float32)/255.0
		images2 = tf.cast(self.raw_images2, tf.float32)/255.0
		count1 = tf.shape(images1)[0]
		count2 = tf.shape(images2)[0]
		images_cat = tf.concat([images1, images2], axis=0)

		layer1 = self._conv_layer('layer1', images_cat, 2, 3, 64) # -> 32x32x64
		layer2 = self._conv_layer('layer2', layer1, 2, 64, 64) # -> 16x16x64
		layer3 = self._conv_layer('layer3', layer2, 2, 64, 64) # -> 8x8x64
		layer4 = self._conv_layer('layer4', layer3, 2, 64, 64) # -> 4x4x64
		layer5 = self._conv_layer('layer5', layer4, 2, 64, 64) # -> 2x2x64
		layer6 = self._conv_layer('layer6', layer5, 2, 64, 64, {'activation': 'none'})[:, 0, 0, :]

		features1 = layer6[0:count1]
		features2 = layer6[count1:count1+count2]
		tile1 = tf.tile(tf.reshape(features1, [count1, 1, 64]), [1, count2, 1])
		tile2 = tf.tile(tf.reshape(features2, [1, count2, 64]), [count1, 1, 1])
		zeros = tf.zeros([count1, count2, 64], dtype=tf.float32)
		pairs = tf.concat([zeros, tile1, tile2], axis=2)

		context = 'longim'
		with tf.variable_scope('matcher' + context, reuse=tf.AUTO_REUSE):
			matcher1 = self._fc_layer('matcher1', pairs, 3*64, 256)
			matcher2 = self._fc_layer('matcher2', matcher1, 256, 65, {'activation': 'none'})

		return matcher2[:, :, 0]

	def __init__(self, reid=False):
		tf.reset_default_graph()

		self.is_training = False
		self.inputs = tf.placeholder(tf.float32, [None, 10])
		self.states = tf.placeholder(tf.float32, [None, 64])
		self.boxes = tf.placeholder(tf.float32, [None, 5])
		num_inputs = tf.shape(self.inputs)[0]
		num_boxes = tf.shape(self.boxes)[0]

		def rnn_step(prev_state, cur_input):
			with tf.variable_scope('rnn_step', reuse=tf.AUTO_REUSE):
				pairs = tf.concat([prev_state, cur_input], axis=1)
				rnn1 = self._fc_layer('rnn1', pairs, 64+10, 128)
				rnn2 = self._fc_layer('rnn2', rnn1, 128, 128)
				rnn3 = self._fc_layer('rnn3', rnn2, 128, 128)
				rnn4 = self._fc_layer('rnn4', rnn3, 128, 128, {'activation': 'none'})
				return rnn4[:, 0:64], rnn4[:, 64:128]

		self.out_states, rnn_outputs = rnn_step(self.states, self.inputs)

		features = tf.concat([rnn_outputs, self.inputs], axis=1)
		features = tf.reshape(features, [num_inputs, 1, 64+10])
		features_tiled = tf.tile(features, [1, num_boxes, 1])
		flat_features_tiled = tf.reshape(features_tiled, [num_inputs*num_boxes, 64+10])
		flat_boxes = tf.reshape(
			tf.tile(
				tf.reshape(self.boxes, [1, num_boxes, 5]),
				[num_inputs, 1, 1]
			),
			[num_inputs*num_boxes, 5]
		)
		fc_cat = tf.concat([flat_features_tiled, flat_boxes], axis=1)
		fc1 = self._fc_layer('fc1', fc_cat, 64+10+5, 64)
		fc2 = self._fc_layer('fc2', fc1, 64, 64)
		fc3 = self._fc_layer('fc3', fc2, 64, 64)
		fc4 = self._fc_layer('fc4', fc3, 64, 1, {'activation': 'none'})
		#fc4 = self._fc_layer('fc4', fc1, 64, 1, {'activation': 'none'})
		self.pre_outputs = tf.reshape(fc4, [num_inputs, num_boxes])
		self.pre_outputs = tf.concat([
			self.pre_outputs[:, 0:num_boxes-1],
			-6*tf.ones([num_inputs, 1], dtype=tf.float32),
		], axis=1)
		self.outputs = tf.nn.softmax(self.pre_outputs)

		if reid:
			reid_logits = self.reid_net()
			self.reid_probs = reid_logits
			self.rnn_probs = self.pre_outputs[:, 0:num_boxes-1]
			#self.reid_probs = tf.nn.softmax(reid_logits)
			#self.rnn_probs = self.outputs
			'''self.outputs = tf.concat([
				#tf.minimum(self.rnn_probs[:, 0:num_boxes-1], self.reid_probs),
				(self.rnn_probs[:, 0:num_boxes-1] + self.reid_probs)/2,
				#self.reid_probs,

				#self.rnn_probs[:, num_boxes-1:num_boxes],
				0.2*tf.ones([num_inputs, 1], dtype=tf.float32),
			], axis=1)
			#self.pre_outputs = tf.minimum(self.pre_outputs, reid_logits)'''

			logits = tf.concat([
				(self.pre_outputs[:, 0:num_boxes-1] + reid_logits)/2,
				#reid_logits,
				-6*tf.ones([num_inputs, 1], dtype=tf.float32),
			], axis=1)
			self.outputs = tf.nn.softmax(logits)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None, var_list=[var for var in tf.global_variables() if var.name.startswith('fc') or var.name.startswith('rnn_step')])
		if reid:
			self.reid_saver = tf.train.Saver(max_to_keep=None, var_list=[var for var in tf.global_variables() if var.name.startswith('layer') or var.name.startswith('matcherlongim')])
