from discoverlib import geom
import model as model

import json
import math
import numpy
import skimage.transform
import struct
import sys
import tensorflow as tf
#tf.disable_eager_execution()
import graph_nets

data_root = sys.argv[1]
label = sys.argv[2]
orig_width = int(sys.argv[3])
orig_height = int(sys.argv[4])

model_path = os.path.join(data_root, 'dataset', label, "tracker-miris-model/model")

NORM = 1000.0
CROP_SIZE = 64

def eprint(s):
	sys.stderr.write(s+"\n")
	sys.stderr.flush()

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

# from apply5b.py
def get_frame_pair(info1, info2, skip):
	senders = []
	receivers = []
	input_nodes = []
	input_edges = []
	input_crops = numpy.zeros((len(info1)+1+len(info2), CROP_SIZE, CROP_SIZE, 3), dtype='uint8')

	for i, t in enumerate(info1):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 1, 0, 0, skip/50.0] + [0.0]*64)
		input_crops[i, :, :, :] = crop
	input_nodes.append([0.5, 0.5, 0, 0, 0, 1, 0, skip/50.0] + [0.0]*64)
	for i, t in enumerate(info2):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 0, 0, 1, skip/50.0] + [0.0]*64)
		input_crops[len(info1)+1+i, :, :, :] = crop

	for i, t1 in enumerate(info1):
		detection1, _, _ = t1
		x1, y1 = get_loc(detection1)

		for j, t2 in enumerate(info2):
			detection2, _, _ = t2
			x2, y2 = get_loc(detection2)

			senders.extend([i, len(info1) + 1 + j])
			receivers.extend([len(info1) + 1 + j, i])
			edge_shared = [x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))]
			input_edges.append(edge_shared + [1, 0, 0])
			input_edges.append(edge_shared + [0, 1, 0])

		senders.extend([i, len(info1)])
		receivers.extend([len(info1), i])
		edge_shared = [0.0, 0.0, 0.0]
		input_edges.append(edge_shared + [0, 0, 0])
		input_edges.append(edge_shared + [1, 0, 0])

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
	add_internal_edges(info1, 0)
	add_internal_edges(info2, len(info1) + 1)

	input_dict = {
		"globals": [],
		"nodes": input_nodes,
		"edges": input_edges,
		"senders": senders,
		"receivers": receivers,
	}
	return input_dict, input_crops

class Detection(object):
	def __init__(self, d, frame_idx, image):
		self.d = d
		self.frame_idx = frame_idx
		self.left = d['left']
		self.top = d['top']
		self.right = d['right']
		self.bottom = d['bottom']
		self.image = image

input_example = {
	'globals': [],
	'nodes': [[1.0] * (8+64), [1.0] * (8+64)],
	'edges': [[1.0] * 6, [1.0] * 6],
	'senders': [0, 1],
	'receivers': [1, 0],
}
target_example = {
	'globals': [],
	'nodes': [[], []],
	'edges': [[0.0], [1.0]],
	'senders': [0, 1],
	'receivers': [1, 0],
}
m = model.Model([[input_example], [target_example]])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
m.saver.restore(session, model_path)

class Tracker(object):
	def __init__(self):
		# for each active object, a list of detections
		self.objects = {}
		# processed frame indexes
		self.frames = []
		# object id counter
		self.next_id = 0

	# returns:
	# (1) track IDs for detections in the current frame
	# (2) the confidence of this frame, i.e., min(1 - 2nd highest score / highest score) taken over active objects
	# (3) if gt set: map from active objects to the minimum threshold needed for this frame
	#	 - if highest prob matched gt: min threshold is 0
	#	 - else: threshold is 1 - (gt score) / (highest score)
	# gt is a map from active object id -> the idx of current frame detections that object should match with
	# if an object id doesn't appear in gt, then we remove it from active set
	def update(self, frame_idx, im, detections, gt=None):
		# cleanup frames that are now in future if needed
		if len(self.frames) > 0 and frame_idx < self.frames[-1]:
			for id in list(self.objects.keys()):
				self.objects[id] = [d for d in self.objects[id] if d.frame_idx < frame_idx]
				if len(self.objects[id]) == 0:
					del self.objects[id]
			self.frames = [idx for idx in self.frames if idx < frame_idx]
		self.frames.append(frame_idx)
		if len(self.frames) >= 2:
			skip = self.frames[-1] - self.frames[-2]
		else:
			skip = 0

		# get info2
		info2 = []
		for idx, d in enumerate(detections):
			sx = clip(d['left']*im.shape[1]//orig_width, 0, im.shape[1])
			sy = clip(d['top']*im.shape[0]//orig_height, 0, im.shape[0])
			ex = clip(d['right']*im.shape[1]//orig_width, 0, im.shape[1])
			ey = clip(d['bottom']*im.shape[0]//orig_height, 0, im.shape[0])
			if ex-sx < 4:
				sx = max(0, sx-2)
				ex = min(im.shape[1], ex+2)
			if ey-sy < 4:
				sy = max(0, sy-2)
				ey = min(im.shape[0], ey+2)

			crop = im[sy:ey, sx:ex, :]
			resize_factor = min([64.0/crop.shape[0], 64.0/crop.shape[1]])
			resize_shape = [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)]
			crop = skimage.transform.resize(crop, resize_shape, preserve_range=True).astype('uint8')

			fix_crop = numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8')
			fix_crop[0:crop.shape[0], 0:crop.shape[1], :] = crop
			d['width'] = float(d['right']-d['left'])/NORM
			d['height'] = float(d['bottom']-d['top'])/NORM
			info2.append((d, fix_crop, idx))

		# helper func to extract matches from a matrix
		def extract_matches(mat):
			votes = {}
			for i in range(mat.shape[0]):
				for j in range(mat.shape[1]):
					p = mat[i, j]
					if i not in votes or p > votes[i][0]:
						if j == mat.shape[1]-1:
							votes[i] = (p, None)
						else:
							votes[i] = (p, j)
			# group by receiver and vote on max idx2->idx1 to eliminate duplicates
			votes2 = {}
			for idx1, t in votes.items():
				p, idx2 = t
				if idx2 is not None and (idx2 not in votes2 or p > votes2[idx2][0]):
					votes2[idx2] = (p, idx1)
			return {idx1: idx2 for (idx2, (_, idx1)) in votes2.items()}

		def get_mat(objects, detections):
			info1 = []
			for idx, (_, obj) in enumerate(objects):
				info1.append((obj[-1].d, obj[-1].image, idx))

			input_dict, input_crops = get_frame_pair(info1, info2, skip)
			d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([input_dict]))
			feed_dict = {
				m.input_crops: input_crops.astype('float32')/255,
				m.is_training: False,
			}
			feed_dict.update(d1)
			outputs = session.run(m.outputs, feed_dict=feed_dict)[:, 0]

			mat = numpy.zeros((len(objects), len(detections)+1), dtype='float32')
			for i, sender in enumerate(input_dict['senders']):
				receiver = input_dict['receivers'][i]
				if sender >= len(info1) or receiver < len(info1):
					continue
				_, _, idx1 = info1[sender]
				output = outputs[i]
				if receiver == len(info1):
					idx2 = len(detections)
				else:
					_, _, idx2 = info2[receiver - len(info1) - 1]
				mat[idx1, idx2] = output

			return mat

		# match each object with current frame
		objects = [(id, dlist) for id, dlist in self.objects.items() if dlist[-1].frame_idx == frame_idx-skip]
		id_to_obj_idx = {}
		obj_idx_to_id = {}
		for i, (id, dlist) in enumerate(objects):
			id_to_obj_idx[id] = i
			obj_idx_to_id[i] = id

		if len(detections) > 0 and len(objects) > 0:
			mat = get_mat(objects, detections)
		else:
			mat = numpy.zeros((len(objects), len(detections)+1), dtype='float32')

		# update objects based either on mat or gt
		min_thresholds = {}
		if gt is None:
			matches = extract_matches(mat)
			track_ids = [None]*len(detections)
			for idx1 in range(len(objects)):
				if idx1 not in matches:
					continue

				idx2 = matches[idx1]
				d = Detection(detections[idx2], frame_idx, info2[idx2][1])
				track_ids[idx2] = obj_idx_to_id[idx1]
				objects[idx1][1].append(d)
			for idx2, d in enumerate(detections):
				if track_ids[idx2] is not None:
					continue
				id = self.next_id
				self.next_id += 1
				d = Detection(d, frame_idx, info2[idx2][1])
				self.objects[id] = [d]
				track_ids[idx2] = id
		else:
			# remove objects not in gt
			for id in list(self.objects.keys()):
				if id not in gt:
					del self.objects[id]

			# update objects
			track_ids = [-1]*len(detections)
			for id, idx2 in gt.items():
				if idx2 is None:
					min_thresholds[id] = 0.0
					continue

				d = detections[idx2]
				if id not in self.objects:
					self.objects[id] = [Detection(d, frame_idx, info2[idx2][1])]
				else:
					self.objects[id].append(Detection(d, frame_idx, info2[idx2][1]))
				track_ids[idx2] = id

				# also compute per-object min thresholds here
				if id not in id_to_obj_idx:
					min_thresholds[id] = 0.0
					continue
				idx1 = id_to_obj_idx[id]
				if numpy.argmax(mat[idx1, :]) == idx2:
					min_thresholds[id] = 0.0
					continue
				min_thresholds[id] = float(1 - (mat[idx1, idx2]+0.01)/(mat[idx1, :].max()+0.01))

		# compute frame confidence
		if mat.shape[0] > 0:
			high1 = mat.max(axis=1)
			mat_ = numpy.copy(mat)
			mat_[numpy.arange(mat_.shape[0]), numpy.argmax(mat_, axis=1)] = 0
			high2 = mat_.max(axis=1)
			track_confidences = 1 - (high2+0.01)/(high1+0.01)
			frame_confidence = float(numpy.min(track_confidences))
		else:
			frame_confidence = 1.0

		return track_ids, frame_confidence, min_thresholds

if __name__ == '__main__':
	stdin = sys.stdin.detach()
	trackers = {}

	def read_im():
		buf = stdin.read(12)
		if not buf:
			return None
		(l, width, height) = struct.unpack('>III', buf)
		buf = stdin.read(l)
		return numpy.frombuffer(buf, dtype='uint8').reshape((height, width, 3))

	def read_json():
		buf = stdin.read(4)
		if not buf:
			return None
		(l,) = struct.unpack('>I', buf)
		buf = stdin.read(l)
		return json.loads(buf.decode('utf-8'))

	while True:
		packet = read_json()
		if packet is None:
			break
		id = packet['id']
		msg = packet['type']

		if msg == 'end':
			del trackers[id]
			continue

		im = read_im()

		detections = packet['detections']
		if detections is None:
			detections = []

		gt = None
		if 'gt' in packet and packet['gt']:
			gt = {int(id): idx for id, idx in packet['gt'].items()}

		if id not in trackers:
			trackers[id] = Tracker()
		track_ids, frame_confidence, min_thresholds = trackers[id].update(packet['frame_idx'], im, detections, gt=gt)
		d = {
			'outputs': track_ids,
			'conf': frame_confidence,
			't': min_thresholds,
		}
		sys.stdout.write('json'+json.dumps(d)+'\n')
		sys.stdout.flush()
