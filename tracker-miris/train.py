# like run4.py but we resize the bounding box to fixed size and add the x/y lengths in gnn

from discoverlib import geom
import model

import graph_nets
import json
import numpy
import os
import math
import random
import skimage.io, skimage.transform
import skvideo.io
import sys
import tensorflow as tf
#tf.disable_eager_execution()
import time

data_root = sys.argv[1]
label = sys.argv[2]
detector = sys.argv[3]

with open(os.path.join(data_root, 'dataset', label, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	orig_dims = cfg['OrigDims']
	cls_set = cfg['Classes']

NORM = 1000.0
CROP_SIZE = 64
SKIPS = [1, 2, 4, 8, 16, 32, 64]

video_path = os.path.join(data_root, 'dataset', label, 'tracker/video/')
json_path = os.path.join(data_root, 'dataset', label, 'tracker/tracks/')
model_path = os.path.join(data_root, 'dataset', label, 'tracker-miris-model/model')

os.makedirs(model_path, exist_ok=True)

labels = []
for fname in os.listdir(video_path):
	if not fname.endswith('.mp4'):
		continue
	label = fname.split('.mp4')[0]
	labels.append(label)

def clip(x, lo, hi):
	if x < lo:
		return lo
	elif x > hi:
		return hi
	else:
		return x

def get_loc(detection):
	cx = (detection['left'] + detection['right']) / 2
	cy = (detection['top'] + detection['bottom']) / 2
	cx = float(cx) / NORM
	cy = float(cy) / NORM
	return cx, cy

def get_frame_pair(info1, info2, skip):
	# we need to get two graphs:
	# * input: (pos, features) for each node, (delta, distance, reverse) for each edge
	# * target: (match) for each edge
	# node <len(info1)> is special, it's for detections in cur frame that don't match any in next frame

	senders = []
	receivers = []
	input_nodes = []
	input_edges = []
	target_edges = []
	input_crops = []

	for i, t in enumerate(info1):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 1, 0, 0, skip/50.0])# + [0.0]*NUM_FEATURES)
		input_crops.append(crop)
	input_nodes.append([0.5, 0.5, 0, 0, 0, 1, 0, skip/50.0])# + [0.0]*NUM_FEATURES)
	input_crops.append(numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8'))
	for i, t in enumerate(info2):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 0, 0, 1, skip/50.0])# + [0.0]*NUM_FEATURES)
		input_crops.append(crop)

	num_matches = 0
	for i, t1 in enumerate(info1):
		detection1, _, _ = t1
		x1, y1 = get_loc(detection1)
		does_match = False

		for j, t2 in enumerate(info2):
			detection2, _, _ = t2
			x2, y2 = get_loc(detection2)

			senders.extend([i, len(info1) + 1 + j])
			receivers.extend([len(info1) + 1 + j, i])
			edge_shared = [x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))]
			input_edges.append(edge_shared + [1, 0, 0])
			input_edges.append(edge_shared + [0, 1, 0])

			if detection1['track_id'] == detection2['track_id']:
				label = 1.0
				does_match = True
			else:
				label = 0.0
			target_edges.append([label])
			target_edges.append([0.0])

		senders.extend([i, len(info1)])
		receivers.extend([len(info1), i])
		edge_shared = [0.0, 0.0, 0.0]
		input_edges.append(edge_shared + [0, 0, 0])
		input_edges.append(edge_shared + [1, 0, 0])

		if does_match:
			label = 0.0
			num_matches += 1
		else:
			label = 1.0
		target_edges.append([label])
		target_edges.append([0.0])

	if num_matches == 0 and False:
		return None, None, None

	def add_internal_edges(info, offset):
		for i, t1 in enumerate(info):
			detection1, _, _ = t1
			x1, y1 = get_loc(detection1)

			for j, t2 in enumerate(info):
				if i == j:
					continue

				detection2, _, _ = t2
				x2, y2 = get_loc(detection2)

				senders.append(offset + i)
				receivers.append(offset + j)
				input_edges.append([x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))] + [0, 0, 1])
				target_edges.append([0.0])
	add_internal_edges(info1, 0)
	add_internal_edges(info2, len(info1) + 1)

	senders = numpy.array(senders, dtype='int32')
	receivers = numpy.array(receivers, dtype='int32')
	input_nodes = numpy.array(input_nodes, dtype='float32')
	input_edges = numpy.array(input_edges, dtype='float32')
	target_edges = numpy.array(target_edges, dtype='float32')

	input_dict = {
		"globals": [],
		"nodes": input_nodes,
		"edges": input_edges,
		"senders": senders,
		"receivers": receivers,
	}
	target_dict = {
		"globals": [],
		"nodes": numpy.zeros((len(input_nodes), 0), dtype='float32'),
		"edges": target_edges,
		"senders": senders,
		"receivers": receivers,
	}
	return input_dict, target_dict, input_crops, num_matches

all_pairs = [] # pairs of frames that are some random skip apart
for label in labels:
	video_fname = os.path.join(video_path, label+'.mp4')
	json_fname = os.path.join(json_path, label+'.json')

	print('... {} load detections'.format(label))
	with open(json_fname, 'r') as f:
		detections = json.load(f)

	# get detection crops (infos)
	print('... {} prepare infos'.format(label))
	infos = {}
	vreader = skvideo.io.vreader(video_fname)
	for frame_idx, im in enumerate(vreader):
		dlist = detections[frame_idx]
		if not dlist:
			continue

		info = []
		for idx, d in enumerate(dlist):
			if d['class'] not in cls_set:
				continue
			d['left'] = clip(d['left']*im.shape[1]//orig_dims[0], 0, im.shape[1])
			d['right'] = clip(d['right']*im.shape[1]//orig_dims[0], 0, im.shape[1])
			d['top'] = clip(d['top']*im.shape[0]//orig_dims[1], 0, im.shape[0])
			d['bottom'] = clip(d['bottom']*im.shape[0]//orig_dims[1], 0, im.shape[0])
			if d['right']-d['left'] < 4 or d['bottom']-d['top'] < 4:
				continue
			crop = im[d['top']:d['bottom'], d['left']:d['right'], :]
			resize_factor = min([float(CROP_SIZE) / crop.shape[0], float(CROP_SIZE) / crop.shape[1]])
			resize_shape = [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)]
			if resize_shape[0] == 0 or resize_shape[1] == 0:
				continue
			crop = skimage.transform.resize(crop, resize_shape, preserve_range=True).astype('uint8')
			fix_crop = numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8')
			fix_crop[0:crop.shape[0], 0:crop.shape[1], :] = crop
			d['width'] = float(d['right']-d['left'])/NORM
			d['height'] = float(d['bottom']-d['top'])/NORM
			info.append((d, fix_crop, idx))

		infos[frame_idx] = info

	print('... {} create examples'.format(label))
	for frame_idx in range(len(detections)):
		skip = random.choice(SKIPS)
		if frame_idx+skip >= len(detections) or not infos.get(frame_idx, None) or not infos.get(frame_idx+skip, None):
			continue
		info1 = infos[frame_idx]
		info2 = infos[frame_idx+skip]
		if len(info1) == 0 or len(info2) == 0:
			continue
		input_dict, target_dict, input_crops, num_matches = get_frame_pair(info1, info2, skip)
		if input_dict is None or target_dict is None:
			continue
		all_pairs.append((
			input_dict, target_dict, input_crops,
			label, frame_idx,
			[get_loc(detection) for detection, _, _ in info1],
			[get_loc(detection) for detection, _, _ in info2],
			num_matches
		))

random.shuffle(all_pairs)
val_pairs = [pair for pair in all_pairs if int(pair[3])%4 == 0]
val_pairs = random.sample(val_pairs, 2048)
train_pairs = [pair for pair in all_pairs if int(pair[3])%4 != 0]

print('initializing model')
m = model.Model([[val_pairs[0][0]], [val_pairs[0][1]]])
session = tf.Session()
session.run(m.init_op)

print('begin training')
best_loss = None

for epoch in range(9999):
	start_time = time.time()
	train_losses = []
	for _ in range(1024):
		batch_examples = random.sample(train_pairs, model.BATCH_SIZE)
		d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[0] for example in batch_examples]))
		d2 = graph_nets.utils_tf.get_feed_dict(m.targets, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[1] for example in batch_examples]))
		feed_dict = {
			m.input_crops: numpy.concatenate([example[2] for example in batch_examples], axis=0).astype('float32')/255,
			m.is_training: True,
			m.learning_rate: 1e-4,
		}
		feed_dict.update(d1)
		feed_dict.update(d2)
		_, loss = session.run([m.optimizer, m.loss], feed_dict=feed_dict)
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in range(0, len(val_pairs), model.BATCH_SIZE):
		batch_examples = val_pairs[i:i+model.BATCH_SIZE]
		d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[0] for example in batch_examples]))
		d2 = graph_nets.utils_tf.get_feed_dict(m.targets, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[1] for example in batch_examples]))
		feed_dict = {
			m.input_crops: numpy.concatenate([example[2] for example in batch_examples], axis=0).astype('float32')/255,
			m.is_training: False,
		}
		feed_dict.update(d1)
		feed_dict.update(d2)
		loss, outputs = session.run([m.loss, m.outputs], feed_dict=feed_dict)
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print('iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss))

	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, model_path)
