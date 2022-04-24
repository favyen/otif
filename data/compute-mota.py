import json
import motmetrics
import numpy
import sys

fname = sys.argv[1]
gt_fname = sys.argv[2]
freq = int(sys.argv[3])

with open(fname, 'r') as f:
	detections = json.load(f)
	if detections is None:
		detections = []
with open(gt_fname, 'r') as f:
	gt_detections = json.load(f)

def interpolate(detections):
	# get tracks
	track_map = {}
	for frame_idx, dlist in enumerate(detections):
		if dlist is None:
			continue
		for detection in dlist:
			detection['frame_idx'] = frame_idx
			track_id = detection['track_id']
			if track_id not in track_map:
				track_map[track_id] = []
			track_map[track_id].append(detection)

	# interpolate tracks
	ndetections = [[] for _ in detections]
	for track in track_map.values():
		ntrack = []
		for detection in track:
			if len(ntrack) > 0:
				prev = ntrack[-1]
				next = detection
				jump = next['frame_idx'] - prev['frame_idx']
				for i in range(1, jump):
					prev_weight = float(jump-i) / float(jump)
					next_weight = float(i) / float(jump)
					interp = {
						'track_id': prev['track_id'],
						'frame_idx': prev['frame_idx']+i,
					}
					for k in ['left', 'top', 'right', 'bottom']:
						interp[k] = int(prev[k]*prev_weight + next[k]*next_weight)
					ntrack.append(interp)
			ntrack.append(detection)

		for detection in ntrack:
			ndetections[detection['frame_idx']].append(detection)

	return ndetections

detections = interpolate(detections)

def to_box(detection):
	return [
		detection['left'],
		detection['top'],
		detection['right'] - detection['left'],
		detection['bottom'] - detection['top'],
	]

acc = motmetrics.MOTAccumulator(auto_id=True)

for frame_idx in range(0, min(len(detections), len(gt_detections)), freq):
	my_boxes = []
	my_ids = []
	gt_boxes = []
	gt_ids = []

	if detections[frame_idx] is not None:
		for detection in detections[frame_idx]:
			my_boxes.append(to_box(detection))
			my_ids.append(detection['track_id'])
	if gt_detections[frame_idx] is not None:
		for detection in gt_detections[frame_idx]:
			gt_boxes.append(to_box(detection))
			gt_ids.append(detection['track_id'])

	dist_mat = motmetrics.distances.iou_matrix(gt_boxes, my_boxes, max_iou=0.5)
	acc.update(gt_ids, my_ids, dist_mat)

mh = motmetrics.metrics.create()
summary = mh.compute(acc, metrics=['mota'], name='acc')
print(summary)
