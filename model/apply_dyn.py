import model_dyn as model
import util

import json
import multiprocessing
import numpy
import os, os.path
import random
import skimage.io, skimage.transform
import subprocess
import sys
import tensorflow as tf
#tf.disable_eager_execution()
import time

def eprint(s):
	sys.stderr.write(s+"\n")
	sys.stderr.flush()

batch_size = int(sys.argv[1])
width = int(sys.argv[2])
height = int(sys.argv[3])
threshold = float(sys.argv[4])
detector_width = int(sys.argv[5])
detector_height = int(sys.argv[6])
detector_sizes = json.loads(sys.argv[7])
model_path = sys.argv[8]

# compute our sizes by rounding down from detector sizes
sizes = []
for w, h in detector_sizes:
	w = w*width//detector_width//32
	h = h*height//detector_height//32
	sizes.append((w, h))
sizes.append((width//32, height//32))

def clip(x, lo, hi):
	if x < lo:
		return lo
	if x > hi:
		return hi
	return x

# get smallest area detector size containing this width/height (measured at detector resolution)
def get_detector_size(w, h):
	best_size = None
	for sz in detector_sizes:
		if sz[0] < w or sz[1] < h:
			continue
		if best_size is None or sz[0]*sz[1] < best_size[0]*best_size[1]:
			best_size = sz
	return best_size

def get_windows(images):
	outputs = session.run(m.outputs, feed_dict={
		m.inputs: images,
		m.is_training: False,
	})
	windows = []
	for i in range(len(images)):
		comps = util.get_windows(outputs[i, :, :] > threshold, sizes=sizes)
		l = []
		for comp in comps:
			def transform(x, y):
				x = x*detector_width*32//width
				y = y*detector_height*32//height
				return (x, y)

			# transform component bounds to detector resolution
			# we also need to increase the window bounds to match a detector size
			sx, sy = transform(comp.rect[0], comp.rect[1])
			ex, ey = transform(comp.rect[2]+1, comp.rect[3]+1)
			cx, cy = (sx+ex)//2, (sy+ey)//2
			w, h = get_detector_size(ex-sx, ey-sy)
			cx = clip(cx, w//2, detector_width-w//2)
			cy = clip(cy, h//2, detector_height-h//2)
			bounds = [cx-w//2, cy-h//2, cx+w//2, cy+h//2]

			cells = []
			for x, y in comp.cells:
				sx, sy = transform(x, y)
				ex, ey = transform(x+1, y+1)
				cells.append([sx, sy, ex, ey])

			l.append({
				'bounds': bounds,
				'cells': cells,
			})
		windows.append(l)
	return windows

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
m = model.Model(width, height)
m.saver.restore(session, model_path)

stdin = sys.stdin.detach()
while True:
	buf = stdin.read(batch_size*detector_width*detector_height*3)
	if not buf:
		break
	ims = numpy.frombuffer(buf, dtype='uint8').reshape((batch_size, detector_height, detector_width, 3))
	windows = get_windows(ims)
	print('json'+json.dumps(windows), flush=True)
