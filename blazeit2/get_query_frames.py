import itertools
import json
import numpy
import os
import skimage.io
import subprocess
import sys

import queries

data_root = sys.argv[1]
query_name = sys.argv[2]
best_detector = sys.argv[3]
out_path = sys.argv[4]

with open(os.path.join(data_root, 'dataset', query_name, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	width = cfg['OrigDims'][0]
	height = cfg['OrigDims'][1]
	gap = cfg['FPS']*5

scores_fname = os.path.join(data_root, 'dataset', query_name, 'blazeit2_model', 'scores.json')
detection_dir = os.path.join(data_root, 'dataset', query_name, 'tracker', best_detector)
video_path = os.path.join(data_root, 'dataset', query_name, 'tracker/video/')

print('load scores')
with open(scores_fname, 'r') as f:
	scores = json.load(f)

print('load detections')
detections = {}
for fname in os.listdir(detection_dir):
	if not fname.endswith('.json'):
		continue
	video_id = int(fname.split('.')[0])
	with open(os.path.join(detection_dir, fname), 'r') as f:
		detections[video_id] = json.load(f)

# starting from highest score, pick frames with at least four detections
# include 5 sec (150 frame) gap between selected frames
scores.sort(key=lambda t: -t[1])

print('pick frames')
ignore_frames = set()
picked_frames = []
detector_count = 0
for i, (location, _) in enumerate(scores):
	parts = location.split('_')
	video_id = int(parts[0].split('.')[0])
	frame_idx = int(parts[1])
	if (video_id, frame_idx) in ignore_frames:
		continue

	if frame_idx >= len(detections[video_id]):
		continue

	detector_count += 1
	dlist = detections[video_id][frame_idx]
	score, _, _ = queries.get_score(query_name, dlist)
	if score == 0:
		continue

	picked_frames.append((video_id, frame_idx))
	for offset in range(-gap, gap):
		ignore_frames.add((video_id, frame_idx+offset))
	if len(picked_frames) >= 50:
		break

print('extract {} images (called detector {} times)'.format(len(picked_frames), detector_count))
picked_frames = set(picked_frames)
picked_video_ids = set([location[0] for location in picked_frames])
counter = 0
for video_id in picked_video_ids:
	FNULL = open(os.devnull, 'w')
	pipe = subprocess.Popen([
		'ffmpeg', '-threads', '2', '-nostdin',
		'-i', '{}/{}.mp4'.format(video_path, video_id),
		'-vf', 'scale={}x{}'.format(width, height),
		'-c:v', 'rawvideo', '-pix_fmt', 'rgb24', '-f', 'rawvideo',
		'-',
	], stdout=subprocess.PIPE, stderr=FNULL)
	for frame_idx in itertools.count():
		buf = pipe.stdout.read(width*height*3)
		if not buf:
			break
		if (video_id, frame_idx) not in picked_frames:
			continue
		im = numpy.zeros((height, width, 3), dtype='uint8')
		im[:, :, :] = numpy.frombuffer(buf, dtype='uint8').reshape((height, width, 3))
		_, dlist, _ = queries.get_score(query_name, detections[video_id][frame_idx])
		for d in dlist:
			im[d['top']:d['bottom'], d['left']:d['left']+2, :] = [255, 0, 0]
			im[d['top']:d['bottom'], d['right']-2:d['right'], :] = [255, 0, 0]
			im[d['top']:d['top']+2, d['left']:d['right'], :] = [255, 0, 0]
			im[d['bottom']-2:d['bottom'], d['left']:d['right'], :] = [255, 0, 0]
		skimage.io.imsave('{}/{}.jpg'.format(out_path, counter), im)
		counter += 1
