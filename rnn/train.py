import model4b as model

import json
import numpy
import os
import math
import random
import sys
import tensorflow as tf
#tf.disable_eager_execution()
import time

NORM = 1000.0

data_root = sys.argv[1]
label = sys.argv[2]

root_path = os.path.join(data_root, 'dataset', label)

with open(os.path.join(root_path, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	skips = cfg['Freqs']
	fps = cfg['FPS']

videos = []
for fname in os.listdir(os.path.join(root_path, 'tracker/video/')):
	if not fname.endswith('.mp4'):
		continue
	id = int(fname.split('.mp4')[0])
	with open(os.path.join(root_path, 'tracker/tracks', '{}.json'.format(id)), 'r') as f:
		detections = json.load(f)
	# get first frame and idx and last frame of all unique tracks
	tracks = {}
	for frame_idx, dlist in enumerate(detections):
		if not dlist:
			continue
		for i, d in enumerate(dlist):
			track_id = d['track_id']
			if track_id not in tracks:
				tracks[track_id] = []
			tracks[track_id].append((frame_idx, i))
	if len(tracks) == 0:
		continue
	tracks = [(track_id, dlist) for track_id, dlist in tracks.items()]
	videos.append((detections, tracks))

def repr_detection(t, d):
	return [
		d['left']/NORM,
		d['top']/NORM,
		d['right']/NORM,
		d['bottom']/NORM,
		float(t)/32.0,
	]

fake_d = {
	'left': -NORM,
	'top': -NORM,
	'right': -NORM,
	'bottom': -NORM,
}

def sample(videos):
	my_skips = skips
	#hard = random.randint(0, 1)
	#if hard == 1:
	#	my_skips = [skip for skip in skips if skip >= 8]

	track_length_category = random.randint(0, 2)
	#skip_rng = random.randint(1, len(my_skips))
	skip_rng = 1
	skip_idx = random.randint(0, len(my_skips)-skip_rng)
	sample_skips = my_skips[skip_idx:skip_idx+skip_rng]

	while True:
		detections, tracks = random.choice(videos)

		if track_length_category == 1:
			tracks = [t for t in tracks if len(t[1])>=fps]
		elif track_length_category == 2:
			tracks = [t for t in tracks if len(t[1])>=2*fps]

		if not tracks:
			continue

		track_id, dlist = random.choice(tracks)
		start_frame, start_idx = random.choice(dlist[0:1+len(dlist)//2])

		# -1 means didn't match to anything
		# -2 means invalid box (will always have 0 probability in the label)
		# we train on -2 instead of masking to simplify the model implementation...
		inputs = numpy.zeros((model.MAX_LENGTH, 10), dtype='float32')
		inputs[0, 0:5] = repr_detection(0, detections[start_frame][start_idx])
		inputs[0, 5:10] = repr_detection(0, detections[start_frame][start_idx])
		boxes = numpy.zeros((model.MAX_LENGTH, model.NUM_BOXES, 5), dtype='float32')
		boxes[:, :, :] = -2
		boxes[:, 0, :] = -1
		targets = numpy.zeros((model.MAX_LENGTH, model.NUM_BOXES), dtype='float32')
		targets[:, 0] = 1
		mask = numpy.zeros((model.MAX_LENGTH,), dtype='float32')

		last_d = detections[start_frame][start_idx]
		last_frame = start_frame
		cur_skip = random.choice(sample_skips) # skip from last input
		frame_idx = start_frame + cur_skip
		i = 0
		while frame_idx < len(detections) and i < model.MAX_LENGTH-1 and frame_idx-last_frame < 20+cur_skip:
			# pre-fill input with fake detection in case we don't find the right one
			inputs[i+1, 0:5] = repr_detection(cur_skip, fake_d)

			dlist = []
			if detections[frame_idx]:
				mask[i] = 1
				dlist = detections[frame_idx]
				#while len(dlist) < model.NUM_BOXES-1:
				#	extras = random.choice(random.choice(videos)[0])
				#	if not extras:
				#		continue
				#	if len(dlist)+len(extras) > model.NUM_BOXES-1:
				#		extras = random.sample(extras, model.NUM_BOXES-1-len(dlist))
				#	dlist += extras
				if len(dlist) > model.NUM_BOXES-1:
					# ok, so we need to sample some boxes
					# but we do want to make sure we always include the ground truth
					good = [d for d in dlist if d['track_id'] == track_id]
					bad = [d for d in dlist if d['track_id'] != track_id]
					dlist = good + random.sample(bad, model.NUM_BOXES-1-len(good))
			for det_idx, d in enumerate(dlist):
				boxes[i, det_idx+1, :] = repr_detection(cur_skip, d)
				if d['track_id'] == track_id:
					inputs[i+1, 0:5] = repr_detection(cur_skip, d)
					last_d = d
					last_frame = frame_idx
					targets[i, 0] = 0
					targets[i, det_idx+1] = 1
			# if the track didn't match to anything in this frame,
			# then input is already ok since we set it -1 above

			inputs[i+1, 5:10] = repr_detection(frame_idx - last_frame, last_d)
			cur_skip = random.choice(sample_skips)
			frame_idx += cur_skip
			i += 1

		if i == 0 or mask.max() == 0:
			continue

		return inputs, boxes, mask, targets

random.shuffle(videos)
num_val = len(videos)//10
val_videos = videos[0:num_val]
train_videos = videos[num_val:]

val_examples = [sample(val_videos) for _ in range(512)]

print('initializing model')
m = model.Model()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
session.run(m.init_op)
latest_path = os.path.join(root_path, 'tracker/rnn/model_latest/model')
best_path = os.path.join(root_path, 'tracker/rnn/model_best/model')

os.makedirs(latest_path, exist_ok=True)
os.makedirs(best_path, exist_ok=True)

print('begin training')
best_loss = None

for epoch in range(9999):
	start_time = time.time()
	train_losses = []
	for _ in range(128):
		examples = [sample(train_videos) for _ in range(model.BATCH_SIZE)]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [example[0] for example in examples],
			m.boxes: [example[1] for example in examples],
			m.mask: [example[2] for example in examples],
			m.targets: [example[3] for example in examples],
			m.learning_rate: 1e-3,
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in range(0, len(val_examples), model.BATCH_SIZE):
		examples = val_examples[i:i+model.BATCH_SIZE]
		loss, outputs = session.run([m.loss, m.outputs], feed_dict={
			m.is_training: False,
			m.inputs: [example[0] for example in examples],
			m.boxes: [example[1] for example in examples],
			m.mask: [example[2] for example in examples],
			m.targets: [example[3] for example in examples],
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print('iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss))

	m.saver.save(session, latest_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)
