import model as model
import util

import json
import numpy
import os
import sys

root_path = sys.argv[1]
prev_resolution = sys.argv[2]
prev_threshold = float(sys.argv[3])
cur_detector = sys.argv[4]

# TODO: duplicate functions
def parse_detector(s):
	cfg = json.loads(s)
	return (cfg['Name'], cfg['Dims'][0], cfg['Dims'][1], cfg['Sizes'])

cur_detector_info = parse_detector(cur_detector)

def parse_resolution(s):
	parts = s.split('_')
	width = int(parts[0])
	height = int(parts[1])
	return (width, height)

def get_detector_cost(detector):
	name, width, height = detector[0:3]
	#detector_dir = '{}-{}x{}'.format(name, width, height)
	#with open('{}/train/{}/speed.txt'.format(root_path, detector_dir), 'r') as f:
	#	return int(f.read().strip())
	area = width*height
	return int(0.067*area+2831)

def get_sizes(width, height):
	sizes = {}
	sizes[(width//32, height//32)] = get_detector_cost(cur_detector_info)
	for w, h in cur_detector_info[3][1:]:
		cost = get_detector_cost(['yolov3', w, h])
		w = w*width//cur_detector_info[1]//32
		h = h*height//cur_detector_info[2]//32
		sizes[(w, h)] = cost
	return sizes

def get_cost(rect, sizes):
	best_cost = None
	for (w, h), cost in sizes.items():
		if w < rect[2]-rect[0]+1 or h < rect[3]-rect[1]+1:
			continue
		if best_cost is None or cost < best_cost:
			best_cost = cost
	return best_cost

with open(os.path.join(root_path, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	cls_set = cfg['Classes'].keys()
	cls = list(cls_set)[0]
	orig_dims = cfg['OrigDims']

# cache the outputs and ground truth at each resolution
resolutions = {}
segtrain_path = os.path.join(root_path, 'train/seg-train/')
for dir in os.listdir(segtrain_path):
	if not os.path.exists(os.path.join(segtrain_path, dir, 'out.npy')):
		continue
	width, height = parse_resolution(dir)
	with open(os.path.join(segtrain_path, dir, 'speed.txt'), 'r') as f:
		runtime = int(f.read().strip())
	outputs = numpy.load(os.path.join(segtrain_path, dir, 'out.npy'))
	examples = []
	for i in range(outputs.shape[0]):
		id = (i*4)+1
		examples.append(util.load_target(os.path.join(segtrain_path, 'images', str(id)+'.json'), cls, (width, height), orig_dims, lenient=True))
	resolutions[dir] = {
		'width': width,
		'height': height,
		'examples': examples,
		'outputs': outputs,
		'runtime': runtime,
		'sizes': get_sizes(width, height),
	}

def res_cost(resolution, threshold):
	res = resolutions[resolution]
	# start with the cost of applying segmentation model on all the test examples
	cost = len(res['examples'])*res['runtime']
	# add up the cost of the windows on which we apply detector
	sizes = list(res['sizes'].keys())
	for output in res['outputs']:
		windows = util.get_windows(output > threshold, sizes=sizes)
		for component in windows:
			cost += get_cost(component.rect, res['sizes'])
	return cost

def res_recall(resolution, threshold):
	res = resolutions[resolution]
	sizes = list(res['sizes'].keys())
	fn = 0
	for i, target in enumerate(res['examples']):
		outputs = res['outputs'][i, :, :]
		bad = False
		for x in range(outputs.shape[1]):
			for y in range(outputs.shape[0]):
				is_gt = target[y, x] > 0.5
				is_out = outputs[y, x] > threshold
				if not is_out and is_gt:
					bad = True
		if bad:
			fn += 1
	return float(len(res['examples'])-fn)/len(res['examples'])

prev_cost = res_cost(prev_resolution, prev_threshold)
target_cost = prev_cost*5//6 # original
#with open('/tmp/experiment.txt', 'r') as f:
#	value = int(f.read().strip())
#	target_cost = prev_cost*(10-value)//10 # experiment
print('prev_cost={}, target={}'.format(prev_cost, target_cost))

# for each resolution, increase threshold until cost is better than target
best_resolution = None
best_threshold = None
best_recall = None
for resolution in resolutions.keys():
	threshold = 1e-5
	cost = res_cost(resolution, threshold)
	while cost > target_cost and threshold <= 1.0:
		threshold *= 2
		cost = res_cost(resolution, threshold)
		print(resolution, threshold, cost, resolutions[resolution]['sizes'], res_recall(resolution, threshold))
	if cost > target_cost:
		continue
	recall = res_recall(resolution, threshold)
	print('res={} computed threshold={} cost={} recall={}'.format(resolution, threshold, cost, recall))
	if best_resolution is None or recall > best_recall:
		best_resolution = resolution
		best_threshold = threshold
		best_recall = recall

if best_resolution:
	print('iter{}_{}'.format(best_resolution, best_threshold))
