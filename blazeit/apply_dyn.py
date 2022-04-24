import model64 as model

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
model_path = sys.argv[7]

def get_windows(images):
	outputs = session.run(m.outputs, feed_dict={
		m.inputs: images,
		m.is_training: False,
	})
	windows = []
	for i in range(len(images)):
		if outputs[i] < threshold:
			windows.append([])
			continue
		windows.append([{
			'bounds': [0, 0, detector_width, detector_height],
			'cells': [[0, 0, detector_width, detector_height]],
		}])
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
