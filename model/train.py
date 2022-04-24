import model as model
import util

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
cls = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
model_path = sys.argv[5]

input_dim = [width, height]
scale = 32
output_dim = [input_dim[0]//scale, input_dim[1]//scale]

examples = []
fnames = [fname for fname in os.listdir(data_path) if fname.endswith('.jpg')]
for i, fname in enumerate(fnames):
	if i % 100 == 0:
		print('{}/{}'.format(i, len(fnames)))
	label = fname.split('.jpg')[0]

	im = skimage.io.imread(os.path.join(data_path, label+'.jpg'))
	resized_im = skimage.transform.resize(im, [input_dim[1], input_dim[0]], preserve_range=True).astype('uint8')
	target = util.load_target(os.path.join(data_path, label+'.json'), cls, input_dim, (im.shape[1], im.shape[0]), lenient=True)
	examples.append((resized_im, target, int(label)))

random.shuffle(examples)
val_examples = [example for example in examples if example[2]%4 == 0]
test_examples = [example for example in examples if example[2]%4 == 1]
train_examples = [example for example in examples if example[2]%4 >= 2]

# train
session = tf.Session()
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

sys.exit(0)

def test():
	# for now get precision/recall on cells at 0.1 threshold
	num_targets = 0
	tp = 0
	fp = 0

	bad_images = 0
	for i, example in enumerate(test_examples):
		outputs = session.run(m.outputs, feed_dict={
			m.inputs: [example[0]],
			m.is_training: False,
		})[0, :, :]

		if i < 64:
			skimage.io.imsave('/home/ubuntu/vis/{}_in.jpg'.format(i), example[0])
			skimage.io.imsave('/home/ubuntu/vis/{}_out.jpg'.format(i), (skimage.transform.resize(outputs, example[0].shape[0:2], order=0, preserve_range=True)*255).astype('uint8'))
			skimage.io.imsave('/home/ubuntu/vis/{}_gt.jpg'.format(i), (skimage.transform.resize(example[1], example[0].shape[0:2], order=0, preserve_range=True)*255).astype('uint8'))

		bad = False
		for x in range(output_dim[0]):
			for y in range(output_dim[1]):
				is_gt = example[1][y, x] > 0.5
				is_out = outputs[y, x] > 0.001
				if is_gt:
					num_targets += 1
				if is_out and is_gt:
					tp += 1
				if is_out and not is_gt:
					fp += 1
				if not is_out and is_gt:
					bad = True
		if bad:
			bad_images += 1
	precision = float(tp)/float(tp+fp)
	recall = float(tp)/float(num_targets)
	print('p={}, r={}'.format(precision, recall))
	print('got {}/{} bad images'.format(bad_images, len(test_examples)))
	print('used {} cells'.format(tp+fp))

# code to get next threshold
sizes = [(2, 2), (5, 5), (13, 8)]

# pre-compute outputs on test examples
test_outputs = []
for example in test_examples:
	output = session.run(m.outputs, feed_dict={
		m.inputs: [example[0]],
		m.is_training: False,
	})[0, :, :]
	test_outputs.append(output)

# estimate the # pixels we need to look at for a given threshold
def estimate_cost(threshold, orig_cell_size=128):
	# start with the cost of applying segmentation model on all the test examples
	cost = len(test_examples)*input_dim[0]*input_dim[1]

	# add up the cost of the windows we need to run at full resolution
	for output in test_outputs:
		windows = util.get_windows(output > threshold, sizes=sizes)
		for component in windows:
			cost += len(component.cells)*orig_cell_size*orig_cell_size

	print('estimate cost={} at threshold={}'.format(cost, threshold))
	return cost

# prev_cost is # pixels examined, both by segmentation model and by object detector
# start at previous threshold and iteratively reduce it until we get the cost we want
def get_next_threshold(prev_cost, prev_threshold):
	threshold = prev_threshold
	cost = prev_cost
	while cost > prev_cost*0.5:
		if threshold >= 1.0:
			return None
		threshold = max(1e-5, threshold*2)
		cost = estimate_cost(threshold)
	return threshold
