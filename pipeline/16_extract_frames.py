# extract individual frames from the training video samples
# along with corresponding detections from best detector
# this will be used for training the segmentation model

import eval
import json
import numpy
import os
import random
import skimage.io
import subprocess
import sys

data_root = sys.argv[1]
label = sys.argv[2]
best_detector = sys.argv[3]

with open(os.path.join(data_root, 'dataset', label, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	orig_dims = cfg['OrigDims']

detection_path = os.path.join(data_root, 'dataset', label, 'train', best_detector)
track_path = os.path.join(data_root, 'dataset', label, 'train/seg-train/best-tracks/')
out_path = os.path.join(data_root, 'dataset', label, 'train/seg-train/images/')
os.makedirs(track_path, exist_ok=True)
os.makedirs(out_path, exist_ok=True)
eval.track(label, detection_path, track_path)

# sample up to 1000 frames containing good track (moving)
# each element is a tuple (video id, frame idx)
good_frames = set()
for fname in os.listdir(track_path):
	id = int(fname.split('.json')[0])
	good_tracks = eval.get_good_tracks(label, track_path+fname)
	for track in good_tracks:
		for _ in range(2):
			frame_idx = random.randint(track[0]['frame_idx'], track[-1]['frame_idx'])
			good_frames.add((id, frame_idx))

if len(good_frames) > 1000:
	good_frames = set(random.sample(good_frames, 1000))
print('got {} frames with track'.format(len(good_frames)))

# get 1000 uniform random frames
video_path = os.path.join(data_root, 'dataset', label, 'train/video/')
all_frames = []
for fname in os.listdir(video_path):
	id = int(fname.split('.mp4')[0])
	with open(os.path.join(detection_path, str(id)+'.json'), 'r') as f:
		nframes = len(json.load(f))
	for frame_idx in range(nframes):
		all_frames.append((id, frame_idx))
for id, frame_idx in random.sample(all_frames, 1000):
	good_frames.add((id, frame_idx))

# read all the videos and save the good frames, along with corresponding detections
print('extracting {} frames'.format(len(good_frames)))
counter = 0
for fname in os.listdir(video_path):
	id = int(fname.split('.mp4')[0])
	with open(os.path.join(detection_path, str(id)+'.json'), 'r') as f:
		detections = json.load(f)
	FNULL = open(os.devnull, 'w')
	pipe = subprocess.Popen([
		'ffmpeg', '-threads', '2', '-nostdin',
		'-i', os.path.join(video_path, str(id)+'.mp4'),
		'-vf', 'scale={}x{}'.format(orig_dims[0], orig_dims[1]),
		'-c:v', 'rawvideo', '-pix_fmt', 'rgb24', '-f', 'rawvideo',
		'-',
	], stdout=subprocess.PIPE, stderr=FNULL)
	for frame_idx in range(len(detections)):
		buf = pipe.stdout.read(orig_dims[0]*orig_dims[1]*3)
		if not buf:
			break
		if (id, frame_idx) not in good_frames:
			continue
		im = numpy.frombuffer(buf, dtype='uint8').reshape((orig_dims[1], orig_dims[0], 3))
		skimage.io.imsave(os.path.join(out_path, str(counter)+'.jpg'), im)
		with open(os.path.join(out_path, str(counter)+'.json'), 'w') as f:
			json.dump(detections[frame_idx], f)
		counter += 1
	pipe.wait()
	FNULL.close()
