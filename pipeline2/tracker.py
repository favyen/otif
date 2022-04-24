from discoverlib import geom

import json
import numpy
import struct
import sys

def eprint(s):
	sys.stderr.write(s+"\n")
	sys.stderr.flush()

class Detection(object):
	def __init__(self, d, frame_idx):
		self.frame_idx = frame_idx
		self.left = d['left']
		self.top = d['top']
		self.right = d['right']
		self.bottom = d['bottom']
		self.cls = d['class']

class Tracker(object):
	def __init__(self):
		# for each active object, a list of detections
		self.objects = {}
		# most recent frame we processed
		self.last_frame = -1
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
		if frame_idx < self.last_frame:
			for id in list(self.objects.keys()):
				self.objects[id] = [d for d in self.objects[id] if d.frame_idx < frame_idx]
				if len(self.objects[id]) == 0:
					del self.objects[id]
		self.last_frame = frame_idx

		# helper func to estimate speed of an object
		def estimate_speed(dlist):
			if len(dlist) == 1:
				return (0, 0)
			last = dlist[-1]
			n = max(5, frame_idx - last.frame_idx)
			target_frame = last.frame_idx - n
			best = None
			for d in dlist[:-1]:
				if best is None or abs(d.frame_idx-target_frame) < abs(best.frame_idx-target_frame):
					best = d
			dx = (last.left-best.left+last.right-best.right)/2.0
			dy = (last.top-best.top+last.bottom-best.bottom)/2.0
			scale = float(frame_idx-last.frame_idx)/float(last.frame_idx-best.frame_idx)
			return (dx*scale, dy*scale)

		# helper func to estimate position of an object at current frame
		def estimate_pos(dlist):
			dx, dy = estimate_speed(dlist)
			last = dlist[-1]
			return geom.Rectangle(
				geom.Point(last.left+dx, last.top+dy),
				geom.Point(last.right+dx, last.bottom+dy),
			)

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

		# match each object with current frame
		objects = list(self.objects.items())
		mat = numpy.zeros((len(objects), len(detections)+1), dtype='float32')
		id_to_obj_idx = {}
		obj_idx_to_id = {}
		for i, (id, dlist) in enumerate(objects):
			id_to_obj_idx[id] = i
			obj_idx_to_id[i] = id
			r1 = estimate_pos(dlist)
			for j, d2 in enumerate(detections):
				r2 = geom.Rectangle(
					geom.Point(d2['left'], d2['top']),
					geom.Point(d2['right'], d2['bottom']),
				)
				iou = r1.iou(r2)
				if d2['class'] != dlist[-1].cls:
					iou = 0
				mat[i, j] = iou
			mat[i, len(detections)] = 0.1

		# update objects based either on mat or gt
		min_thresholds = {}
		if gt is None:
			matches = extract_matches(mat)
			track_ids = [None]*len(detections)
			for idx1, idx2 in matches.items():
				d = detections[idx2]
				objects[idx1][1].append(Detection(d, frame_idx))
				track_ids[idx2] = obj_idx_to_id[idx1]
			for idx2, d in enumerate(detections):
				if track_ids[idx2] is not None:
					continue
				id = self.next_id
				self.next_id += 1
				self.objects[id] = [Detection(d, frame_idx)]
				track_ids[idx2] = id

			# remove old objects
			for id in list(self.objects.keys()):
				age = frame_idx - self.objects[id][-1].frame_idx
				if age < 20:
					continue
				del self.objects[id]
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
					self.objects[id] = [Detection(d, frame_idx)]
				else:
					self.objects[id].append(Detection(d, frame_idx))
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
