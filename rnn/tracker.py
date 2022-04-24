from discoverlib import geom
import model_infer4b as model

import json
import numpy
import skimage.transform
import struct
import sys
import tensorflow as tf
#tf.disable_eager_execution()

# run tracker on CPU
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_root = sys.argv[1]
label = sys.argv[2]
orig_width = int(sys.argv[3])
orig_height = int(sys.argv[4])

model_path = os.path.join(data_root, 'dataset', label, 'tracker/rnn/model_best/model')
reid_path = None
reid_enabled = False
if label in ['shibuya', 'warsaw', 'uav']:
	reid_enabled = True
	reid_path = os.path.join(data_root, 'dataset', label, 'tracker/rnn/reid/model')
else:
	reid_enabled = False
	reid_path = None

def eprint(s):
	sys.stderr.write(s+"\n")
	sys.stderr.flush()

NORM = 1000.0
NUM_HIDDEN = 64
def repr_detection(t, d):
	return [
		d['left']/NORM,
		d['top']/NORM,
		d['right']/NORM,
		d['bottom']/NORM,
		float(t)/32.0,
	]

def clip(x, lo, hi):
	if x < lo:
		return lo
	elif x > hi:
		return hi
	else:
		return x

fake_d = {
	'left': -NORM,
	'top': -NORM,
	'right': -NORM,
	'bottom': -NORM,
}

ttt = {
	'read1': 0,
	'read2': 0,
	'read3': 0,
	'cleanup': 0,
	'preproc': 0,
	'match': 0,
	'update': 0,
	'conf': 0,
	'write': 0,
	'count': 0,
}
import time

class Detection(object):
	def __init__(self, d, frame_idx, hidden, last_real=None, image=None):
		self.d = d
		self.frame_idx = frame_idx
		self.hidden = hidden
		self.left = d['left']
		self.top = d['top']
		self.right = d['right']
		self.bottom = d['bottom']
		self.image = image
		if last_real is None:
			self.last_real = self
		else:
			self.last_real = last_real


m = model.Model(reid=reid_enabled)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
m.saver.restore(session, model_path)
if reid_enabled:
	m.reid_saver.restore(session, reid_path)

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
		t0 = time.time()

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

		ttt['cleanup'] += time.time()-t0
		t0 = time.time()

		# get images2 if reid
		images2 = numpy.zeros((len(detections), 64, 64, 3), dtype='uint8')
		if reid_enabled:
			for i, d in enumerate(detections):
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

				images2[i, 0:crop.shape[0], 0:crop.shape[1], :] = crop

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
			inputs = numpy.zeros((len(objects), 10), dtype='float32')
			states = numpy.zeros((len(objects), NUM_HIDDEN), dtype='float32')
			for i, (_, obj) in enumerate(objects):
				if len(obj) >= 2:
					obj_skip = obj[-1].frame_idx - obj[-2].frame_idx
				else:
					obj_skip = 0
				inputs[i, 0:5] = repr_detection(obj_skip, obj[-1].d)
				inputs[i, 5:10] = repr_detection(obj[-1].frame_idx - obj[-1].last_real.frame_idx, obj[-1].last_real.d)
				states[i, :] = obj[-1].hidden
			boxes = numpy.zeros((len(detections)+1, 5), dtype='float32')
			for i, d in enumerate(detections):
				boxes[i, :] = repr_detection(skip, d)
			boxes[len(detections), :] = -1

			feed_dict = {
				m.inputs: inputs,
				m.states: states,
				m.boxes: boxes,
			}

			# handle reid network data
			if reid_enabled:
				images1 = numpy.zeros((len(objects), 64, 64, 3), dtype='uint8')
				for i, (_, obj) in enumerate(objects):
					images1[i, :, :, :] = obj[-1].last_real.image

				bbox2 = numpy.zeros((len(detections), 4), dtype='int32')
				for i, d in enumerate(detections):
					sx = clip(d['left']*im.shape[1]//orig_width, 0, im.shape[1])
					sy = clip(d['top']*im.shape[0]//orig_height, 0, im.shape[0])
					ex = clip(d['right']*im.shape[1]//orig_width, 0, im.shape[1])
					ey = clip(d['bottom']*im.shape[0]//orig_height, 0, im.shape[0])
					while ex-sx < 4:
						sx = max(0, sx-2)
						ex = min(im.shape[1], ex+2)
					while ey-sy < 4:
						sy = max(0, sy-2)
						ey = min(im.shape[0], ey+2)
					bbox2[i] = [sx, sy, ex, ey]

				feed_dict[m.raw_images1] = images1
				#feed_dict[m.raw_frame2] = im
				#feed_dict[m.bbox2] = bbox2
				feed_dict[m.raw_images2] = images2

				mat, hiddens = session.run([m.outputs, m.out_states], feed_dict=feed_dict)
				'''if len(objects) > 0:
					eprint('pid={} inputs={} states={} boxes={} raw1={} raw2={} bbox2={}, mins={} {} min/max={} {}'.format(os.getpid(), inputs.shape, states.shape, boxes.shape, images1.shape, im.shape, bbox2.shape, (bbox2[:, 2]-bbox2[:, 0]).min(), (bbox2[:, 3]-bbox2[:, 1]).min(), bbox2.min(), bbox2.max()))
					mat, hiddens, images2 = session.run([m.outputs, m.out_states, m.raw_images2], feed_dict=feed_dict)
				else:
					mat = numpy.zeros((0, len(detections)+1), dtype='float32')
					hiddens = numpy.zeros((0, NUM_HIDDEN), dtype='float32')
					images2 = session.run(m.raw_images2, feed_dict=feed_dict)'''
			else:
				#mat, hiddens = session.run([m.outputs, m.out_states], feed_dict=feed_dict)
				mat, hiddens = session.run([m.outputs, m.out_states], feed_dict=feed_dict)
				#images2 = numpy.zeros((len(detections), 64, 64, 3), dtype='uint8')

			if frame_idx == 1 and False:
				eprint('=== {} ==='.format(frame_idx))
				eprint(str(detections))
				eprint(str(reid_probs))
				eprint(str(rnn_probs))
				eprint(str(inputs[1]))
				eprint(str(states[1]))
				eprint(str(boxes[6]))
			#eprint('=== {} ==='.format(frame_idx))
			#eprint(str(inputs))
			#eprint(str(boxes))
			#eprint(str(mat))
			return mat, hiddens#, images2

		# match each object with current frame
		ttt['preproc'] += time.time()-t0
		t0 = time.time()
		objects = [(id, dlist) for id, dlist in self.objects.items() if (frame_idx - dlist[-1].last_real.frame_idx) < 20+skip]
		id_to_obj_idx = {}
		obj_idx_to_id = {}
		for i, (id, dlist) in enumerate(objects):
			id_to_obj_idx[id] = i
			obj_idx_to_id[i] = id

		ttt['match'] += time.time()-t0
		t0 = time.time()
		if len(objects) > 0:
			#mat, hiddens, images2 = get_mat(objects, detections)
			mat, hiddens = get_mat(objects, detections)
		else:
			mat = numpy.zeros((len(objects), len(detections)+1), dtype='float32')
			hiddens = numpy.zeros((len(objects), NUM_HIDDEN), dtype='float32')

		# update objects based either on mat or gt
		ttt['update'] += time.time()-t0
		t0 = time.time()
		min_thresholds = {}
		if gt is None:
			matches = extract_matches(mat)
			track_ids = [None]*len(detections)
			for idx1 in range(len(objects)):
				if idx1 in matches:
					idx2 = matches[idx1]
					d = Detection(detections[idx2], frame_idx, hiddens[idx1, :], image=images2[idx2])
					track_ids[idx2] = obj_idx_to_id[idx1]
				else:
					d = Detection(fake_d, frame_idx, hiddens[idx1, :], last_real=objects[idx1][1][-1].last_real)
				objects[idx1][1].append(d)
			for idx2, d in enumerate(detections):
				if track_ids[idx2] is not None:
					continue
				id = self.next_id
				self.next_id += 1
				d = Detection(d, frame_idx, numpy.zeros((NUM_HIDDEN,), dtype='float32'), image=images2[idx2])
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
				if id not in id_to_obj_idx:
					self.objects[id] = []
					hidden = numpy.zeros((NUM_HIDDEN,), dtype='float32')
					last_real = None
				else:
					idx1 = id_to_obj_idx[id]
					hidden = hiddens[idx1, :]
					last_real = self.objects[id][-1].last_real

				if idx2 is None:
					d = Detection(fake_d, frame_idx, hidden, last_real=last_real)
					if last_real is None:
						d.image = numpy.zeros((64, 64, 3), dtype='uint8')
				else:
					d = Detection(detections[idx2], frame_idx, hidden, image=images2[idx2])
					track_ids[idx2] = id
				self.objects[id].append(d)

				# also compute per-object min thresholds here
				if idx2 is None or id not in id_to_obj_idx:
					min_thresholds[id] = 0.0
					continue
				idx1 = id_to_obj_idx[id]
				if numpy.argmax(mat[idx1, :]) == idx2:
					min_thresholds[id] = 0.0
					continue
				min_thresholds[id] = float(1 - (mat[idx1, idx2]+0.01)/(mat[idx1, :].max()+0.01))

		# compute frame confidence
		ttt['match'] += time.time()-t0
		t0 = time.time()
		if mat.shape[0] > 0:
			high1 = mat.max(axis=1)
			mat_ = numpy.copy(mat)
			mat_[numpy.arange(mat_.shape[0]), numpy.argmax(mat_, axis=1)] = 0
			high2 = mat_.max(axis=1)
			track_confidences = 1 - (high2+0.01)/(high1+0.01)
			frame_confidence = float(numpy.min(track_confidences))
		else:
			frame_confidence = 1.0

		ttt['conf'] += time.time()-t0
		t0 = time.time()
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
		t0 = time.time()
		packet = read_json()
		if packet is None:
			break
		id = packet['id']
		msg = packet['type']

		if msg == 'end':
			del trackers[id]
			continue

		ttt['read2'] += time.time()-t0
		t0 = time.time()
		im = read_im()
		ttt['read3'] += time.time()-t0
		t0 = time.time()

		detections = packet['detections']
		if detections is None:
			detections = []

		gt = None
		if 'gt' in packet and packet['gt']:
			gt = {int(id): idx for id, idx in packet['gt'].items()}

		if id not in trackers:
			trackers[id] = Tracker()
		ttt['read3'] += time.time()-t0
		track_ids, frame_confidence, min_thresholds = trackers[id].update(packet['frame_idx'], im, detections, gt=gt)
		t0 = time.time()
		d = {
			'outputs': track_ids,
			'conf': frame_confidence,
			't': min_thresholds,
		}
		sys.stdout.write('json'+json.dumps(d)+'\n')
		sys.stdout.flush()
		ttt['write'] += time.time()-t0
		ttt['count'] += 1
		if ttt['count']%1000==0:
			eprint(str(ttt))
