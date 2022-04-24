import model64 as model

import json
import numpy
import os, os.path
import random
import skimage.io, skimage.transform
import sys
import tensorflow as tf
#tf.disable_eager_execution()
import time

data_path = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
model_path = sys.argv[4]

input_dim = [width, height]

examples = []
fnames = [fname for fname in os.listdir(data_path) if fname.endswith('.jpg')]
for i, fname in enumerate(fnames):
	if i % 100 == 0:
		print('{}/{}'.format(i, len(fnames)))
	label = fname.split('.jpg')[0]

	im = skimage.io.imread(os.path.join(data_path, label+'.jpg'))
	resized_im = skimage.transform.resize(im, [input_dim[1], input_dim[0]], preserve_range=True).astype('uint8')
	with open(os.path.join(data_path, label+'.txt'), 'r') as f:
		target = int(f.read().strip())
	examples.append((resized_im, target, int(label)))

random.shuffle(examples)
val_examples = [example for example in examples if example[2]%4 == 0]
test_examples = [example for example in examples if example[2]%4 == 1]
train_examples = [example for example in examples if example[2]%4 >= 2]

# train
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
m = model.Model(input_dim[0], input_dim[1])
session.run(m.init_op)
batch_size = 32
best_loss = None
bad_loss_streak = 0
for epoch in range(9999):
	start_time = time.time()
	train_losses = []
	random.shuffle(train_examples)
	for i in range(0, len(train_examples), batch_size):
		batch = train_examples[i:i+batch_size]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.inputs: [example[0] for example in batch],
			m.targets: [example[1] for example in batch],
			m.learning_rate: 1e-3,
			m.is_training: True,
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in range(0, len(val_examples), batch_size):
		batch = val_examples[i:i+batch_size]
		loss = session.run(m.loss, feed_dict={
			m.inputs: [example[0] for example in batch],
			m.targets: [example[1] for example in batch],
			m.is_training: False,
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print(
		'iteration ({}, {}) {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(
		width, height, epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss
	))

	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, model_path)
		bad_loss_streak = 0
	else:
		bad_loss_streak += 1
		if bad_loss_streak > 25:
			break
