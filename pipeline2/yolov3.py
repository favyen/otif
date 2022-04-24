import json
import numpy
import os
import shutil
import sys

data_root = sys.argv[1]
batch_size = int(sys.argv[2])
width = int(sys.argv[3])
height = int(sys.argv[4])
param_width = int(sys.argv[5])
param_height = int(sys.argv[6])
threshold = float(sys.argv[7])
classes = sys.argv[8]
label = sys.argv[9]

os.chdir(os.path.join(data_root, 'darknet-alexey/'))
sys.path.append('./')
import darknet

def eprint(s):
	sys.stderr.write(str(s) + "\n")
	sys.stderr.flush()

if classes != '':
	classes = {cls.strip(): True for cls in classes.split(',')}
else:
	classes = None

detector_label = label
if detector_label.startswith('caldot'):
	detector_label = 'caldot'
if detector_label in ['amsterdam', 'jackson']:
	detector_label = 'generic'

config_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3-{}x{}-test.cfg'.format(param_width, param_height))
meta_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.data')
names_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.names')

if detector_label == 'generic':
	weight_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3.best')
else:
	weight_path = os.path.join(data_root, 'yolov3', label, 'yolov3-{}x{}.best'.format(param_width, param_height))

# ensure width/height in config file
with open(config_path, 'r') as f:
	tmp_config_buf = ''
	for line in f.readlines():
		line = line.strip()
		if line.startswith('width='):
			line = 'width={}'.format(width)
		if line.startswith('height='):
			line = 'height={}'.format(height)
		tmp_config_buf += line + "\n"
tmp_config_path = '/tmp/yolov3-{}.cfg'.format(os.getpid())
with open(tmp_config_path, 'w') as f:
	f.write(tmp_config_buf)

# Write out our own obj.data which has direct path to obj.names.
tmp_obj_names = '/tmp/obj-{}.names'.format(os.getpid())
shutil.copy(names_path, tmp_obj_names)

with open(meta_path, 'r') as f:
	tmp_meta_buf = ''
	for line in f.readlines():
		line = line.strip()
		if line.startswith('names='):
			line = 'names={}'.format(tmp_obj_names)
		tmp_meta_buf += line + "\n"
tmp_obj_meta = '/tmp/obj-{}.data'.format(os.getpid())
with open(tmp_obj_meta, 'w') as f:
	f.write(tmp_meta_buf)

# Finally we can load YOLOv3.
net, class_names, _ = darknet.load_network(tmp_config_path, meta_path, weight_path, batch_size=batch_size)
os.remove(tmp_config_path)
os.remove(tmp_obj_names)
os.remove(tmp_obj_meta)

stdin = sys.stdin.detach()
while True:
	buf = stdin.read(batch_size*width*height*3)
	if not buf:
		break

	arr = numpy.frombuffer(buf, dtype='uint8').reshape((batch_size, height, width, 3))
	arr = arr.transpose((0, 3, 1, 2))
	arr = numpy.ascontiguousarray(arr.flat, dtype='float32')/255.0
	darknet_images = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
	darknet_images = darknet.IMAGE(width, height, 3, darknet_images)
	raw_detections = darknet.network_predict_batch(net, darknet_images, batch_size, width, height, threshold, 0.5, None, 0, 0)
	detections = []
	for idx in range(batch_size):
		num = raw_detections[idx].num
		raw_dlist = raw_detections[idx].dets
		darknet.do_nms_obj(raw_dlist, num, len(class_names), 0.45)
		raw_dlist = darknet.remove_negatives(raw_dlist, class_names, num)
		dlist = []
		for cls, score, (cx, cy, w, h) in raw_dlist:
			if classes is not None and cls not in classes:
				continue
			dlist.append({
				'class': cls,
				'score': float(score),
				'left': int(cx-w/2),
				'right': int(cx+w/2),
				'top': int(cy-h/2),
				'bottom': int(cy+h/2),
			})
		detections.append(dlist)
	darknet.free_batch_detections(raw_detections, batch_size)
	print('json'+json.dumps(detections), flush=True)
