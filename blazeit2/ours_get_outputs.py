import itertools
import json
import math
import numpy
import os
import skimage.io
import sys
import subprocess

import queries

data_root = sys.argv[1]
query_name = sys.argv[2]
track_path = sys.argv[3]
out_path = sys.argv[4]

with open(os.path.join(data_root, 'dataset', query_name, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	width = cfg['OrigDims'][0]
	height = cfg['OrigDims'][1]
	gap = cfg['FPS']*5

video_path = os.path.join(data_root, 'dataset', query_name, 'tracker/video/')

scores = []
frame_to_detections = {}

for fname in os.listdir(track_path):
	video_id = int(fname.split('.')[0])

	with open(os.path.join(track_path, fname), 'r') as f:
		detections = json.load(f)

	if not detections:
		continue

	# get the displacement of each track
	tracks = {}
	for dlist in detections:
		if not dlist:
			continue
		for d in dlist:
			track_id = d['track_id']
			if track_id not in tracks:
				tracks[track_id] = []
			tracks[track_id].append(d)
	track_displacements = {}
	track_lengths = {}
	for track_id, track in tracks.items():
		sx = (track[0]['left']+track[0]['right'])//2
		sy = (track[0]['top']+track[0]['bottom'])//2
		ex = (track[-1]['left']+track[-1]['right'])//2
		ey = (track[-1]['top']+track[-1]['bottom'])//2
		track_displacements[track_id] = math.sqrt((ex-sx)**2+(ey-sy)**2)
		track_lengths[track_id] = len(track)

	# compute scores on each frame
	for frame_idx, dlist in enumerate(detections):
		if not dlist:
			continue
		'''dlist = [d for d in dlist if (d['top']+d['bottom'])//2 > 400 and d['class'] == 'car']
		if len(dlist) < 4:
			continue
		cur_displacements = []
		for d in dlist:
			cur_displacements.append(track_lengths[d['track_id']])
		cur_displacements.sort()
		score = min(cur_displacements[-4:])'''
		score, dlist, _ = queries.get_score(query_name, dlist, track_lengths)
		if score == 0:
			continue
		scores.append((video_id, frame_idx, score))
		frame_to_detections[(video_id, frame_idx)] = dlist

scores.sort(key=lambda t: -t[2])

print('pick frames')
ignore_frames = set()
picked_frames = {}
for video_id, frame_idx, score in scores:
	if (video_id, frame_idx) in ignore_frames:
		continue

	picked_frames[(video_id, frame_idx)] = len(picked_frames)
	print(video_id, frame_idx, score)
	for offset in range(-gap, gap):
		ignore_frames.add((video_id, frame_idx+offset))
	if len(picked_frames) >= 50:
		break

print('extract {} images'.format(len(picked_frames)))
picked_video_ids = set([location[0] for location in picked_frames.keys()])
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
		output_idx = picked_frames[(video_id, frame_idx)]
		im = numpy.zeros((height, width, 3), dtype='uint8')
		im[:, :, :] = numpy.frombuffer(buf, dtype='uint8').reshape((height, width, 3))
		for d in frame_to_detections[(video_id, frame_idx)]:
			im[d['top']:d['bottom'], d['left']:d['left']+2, :] = [255, 0, 0]
			im[d['top']:d['bottom'], d['right']-2:d['right'], :] = [255, 0, 0]
			im[d['top']:d['top']+2, d['left']:d['right'], :] = [255, 0, 0]
			im[d['bottom']-2:d['bottom'], d['left']:d['right'], :] = [255, 0, 0]
		skimage.io.imsave('{}/{}.jpg'.format(out_path, output_idx), im)
