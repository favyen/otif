import model as model

import json
import numpy
import os, os.path
import random
import skimage.io, skimage.transform
import sys
import tensorflow as tf
#tf.disable_eager_execution()
import time

data_path = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
model_path = sys.argv[4]
out_path = sys.argv[5]

session = tf.Session()
m = model.Model(width, height)
m.saver.restore(session, model_path)

fnames = [fname for fname in os.listdir(data_path) if fname.endswith('.jpg')]
examples = []
for fname in fnames:
	id = int(fname.split('.jpg')[0])
	if id%4 != 1:
		continue
	im = skimage.io.imread(os.path.join(data_path, fname))
	im = skimage.transform.resize(im, [height, width], preserve_range=True).astype('uint8')
	examples.append((id, im))
examples.sort(key=lambda x: x[0])

batch_size = 64
session.run(m.outputs, feed_dict={
	m.inputs: [example[1] for example in examples[0:batch_size]],
	m.is_training: False,
})

t0 = time.time()
outputs = numpy.zeros((len(examples), height//32, width//32), dtype='float32')
for i in range(0, len(examples), batch_size):
	start, end = i, min(i+batch_size, len(examples))
	outputs[start:end, :, :] = session.run(m.outputs, feed_dict={
		m.inputs: [example[1] for example in examples[start:end]],
		m.is_training: False,
	})

numpy.save(os.path.join(out_path, 'out.npy'), outputs)

ms_taken = int(1000*(time.time() - t0))
ms_per_sample = ms_taken*30*60//len(examples)
with open(os.path.join(out_path, 'speed.txt'), 'w') as f:
	f.write(str(ms_per_sample))
