import model64 as model

import itertools
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

data_root = sys.argv[1]
label = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
model_path = sys.argv[5]
out_path = sys.argv[6]

batch_size = 32

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
m = model.Model(width, height)
m.saver.restore(session, model_path)

root_path = os.path.join(data_root, 'dataset', label)
video_path = os.path.join(root_path,  'tracker/video/')
scores = []
for fname in os.listdir(video_path):
	FNULL = open(os.devnull, 'w')
	pipe = subprocess.Popen([
		'ffmpeg', '-threads', '2', '-nostdin',
		'-i', os.path.join(video_path, fname),
		'-vf', 'scale={}x{}'.format(width, height),
		'-c:v', 'rawvideo', '-pix_fmt', 'rgb24', '-f', 'rawvideo',
		'-',
	], stdout=subprocess.PIPE, stderr=FNULL)
	cur_batch = []
	for frame_idx in itertools.count():
		buf = pipe.stdout.read(width*height*3)
		if not buf:
			break
		im = numpy.frombuffer(buf, dtype='uint8').reshape((height, width, 3))
		cur_batch.append(im)

		if len(cur_batch) >= batch_size:
			outputs = session.run(m.outputs, feed_dict={
				m.inputs: cur_batch,
				m.is_training: False,
			})
			del cur_batch[:]
			for i, score in enumerate(outputs.tolist()):
				label = '{}_{}'.format(fname, frame_idx-len(cur_batch)+i)
				scores.append((label, score))
				print(label, score)

with open(out_path, 'w') as f:
	json.dump(scores, f)
