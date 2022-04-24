# visualize export input

import json
import numpy
import os
import skimage.io
import skvideo.io
import sys

export_path = sys.argv[1]

labels = []
for fname in os.listdir(export_path):
	if not fname.endswith('_0.mp4'):
		continue
	label = fname.split('_0.mp4')[0]
	labels.append(label)

for label in labels:
	video_fname = export_path + label + '_0.mp4'
	json_fname = export_path + label + '_1.json'

	with open(json_fname, 'r') as f:
		detections = json.load(f)

	vreader = skvideo.io.vreader(video_fname)
	for frame_idx, im in enumerate(vreader):
		if frame_idx >= len(detections):
			continue
		im = numpy.copy(im)
		print(label, frame_idx)
		dlist = detections[frame_idx]
		if dlist:
			for d in dlist:
				left, top, right, bottom = d['left'], d['top'], d['right'], d['bottom']
				im[top:bottom, left-2:left+2, :] = [255, 0, 0]
				im[top:bottom, right-2:right+2, :] = [255, 0, 0]
				im[top-2:top+2, left:right, :] = [255, 0, 0]
				im[bottom-2:bottom+2, left:right, :] = [255, 0, 0]
		skimage.io.imsave('/home/ubuntu/vis/{}_{}.jpg'.format(label, frame_idx), im)
