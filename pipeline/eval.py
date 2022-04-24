import json
import math
import multiprocessing
import numpy
import os
import random
import shutil
import subprocess

def track_one(args):
	in_fname, out_fname, cls, threshold = args
	subprocess.call(['go', 'run', 'iou-tracker.go', in_fname, out_fname, cls, str(threshold)])

def track(label, in_path, out_path):
	fnames = os.listdir(in_path)
	if label in ['amsterdam', 'jackson', 'shibuya', 'caldot1', 'caldot2', 'warsaw', 'uav']:
		cls = 'car'
	elif label == 'taipei':
		cls = 'bus'
	args = [(os.path.join(in_path, fname), os.path.join(out_path, fname), cls, 0) for fname in fnames if fname.endswith('.json')]
	p = multiprocessing.Pool(16)
	p.map(track_one, args)
	p.close()

def track_and_eval(label, in_path, out_path=None):
	if out_path is None:
		actual_out_path = '/tmp/{}/'.format(random.randint(10000000, 99999999))
		os.mkdir(actual_out_path)
	else:
		actual_out_path = out_path
	track(label, in_path, actual_out_path)
	acc = eval(label, actual_out_path)
	if out_path is None:
		shutil.rmtree(actual_out_path)
	return acc

def get_good_tracks(label, fname):
	with open(fname) as f:
		detections = json.load(f)
	if detections is None:
		detections = []

	track_dict = {}
	for frame_idx in range(len(detections)):
		if not detections[frame_idx]:
			continue
		for detection in detections[frame_idx]:
			detection['frame_idx'] = frame_idx
			track_id = detection['track_id']
			if track_id not in track_dict:
				track_dict[track_id] = []
			track_dict[track_id].append(detection)
	tracks = track_dict.values()

	def get_center(d):
		return (d['left']+d['right'])//2, (d['top']+d['bottom'])//2

	def get_distance(d1, d2):
		x1, y1 = get_center(d1)
		x2, y2 = get_center(d2)
		dx = x1-x2
		dy = y1-y2
		return math.sqrt(dx*dx+dy*dy)

	good_tracks = []
	for track in tracks:
		duration = track[-1]['frame_idx'] - track[0]['frame_idx']
		distance = get_distance(track[0], track[-1])
		if label in ['amsterdam', 'jackson', 'shibuya', 'caldot1', 'caldot2', 'warsaw', 'uav'] and (duration < 10 or distance < 50):
			continue
		elif label == 'taipei' and duration < 30:
			continue
		good_tracks.append(track)

	return good_tracks

def eval(label, path):
	# load ground truth
	gt_fname = '/data2/blazeit/multiscope-test-set/{}/valid/gt.txt'.format(label)
	gt_counts = {}
	with open(gt_fname, 'r') as f:
		for line in f:
			parts = line.strip().split('\t')
			if len(parts) != 2:
				continue
			id = int(parts[0])
			count = int(parts[1])
			gt_counts[id] = count

	accs = []
	for id in gt_counts.keys():
		good_tracks = get_good_tracks(label, path+str(id)+'.json')
		count = len(good_tracks)
		if count == gt_counts[id]:
			acc = 1.0
		elif count == 0 or gt_counts[id] == 0:
			acc = 0.0
		else:
			acc = float(min(count, gt_counts[id])) / max(count, gt_counts[id])
		print(id, acc, count, gt_counts[id])
		accs.append(acc)

	return numpy.mean(accs)
