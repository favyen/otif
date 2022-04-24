import json
import numpy
import os
import sys

root_path = sys.argv[1]
prev_threshold = float(sys.argv[2])
cur_detector = sys.argv[3]

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
	detector_dir = '{}-{}x{}'.format(name, width, height)
	with open(os.path.join(root_path, 'train', detector_dir, 'speed.txt'), 'r') as f:
		return int(f.read().strip())

cur_detector_cost = get_detector_cost(cur_detector_info)

# cache the outputs and ground truth at each resolution
train_out_path = os.path.join(root_path, 'train/blazeit-train/out/')
with open(train_out_path+'/speed.txt', 'r') as f:
	runtime = int(f.read().strip())
outputs = numpy.load(train_out_path+'/out.npy')
examples = []
for i in range(outputs.shape[0]):
	id = (i*4)+1
	with open(os.path.join(root_path, 'train/blazeit-train/images/', str(id)+'.txt'), 'r') as f:
		target = int(f.read().strip())
	examples.append(target)
info = {
	'examples': examples,
	'outputs': outputs,
	'runtime': runtime,
}

def compute_cost(threshold):
	# start with the cost of applying segmentation model on all the test examples
	cost = len(info['examples'])*info['runtime']
	# add up the cost of the windows on which we apply detector
	for output in info['outputs']:
		if output > threshold:
			cost += cur_detector_cost
	return cost

prev_cost = compute_cost(prev_threshold)
if prev_threshold < 0:
	#target_cost = prev_cost
	print('iter64_64_0')
	exit()
else:
	target_cost = prev_cost*9//10
print('prev_cost={}, target={}'.format(prev_cost, target_cost))

# increase threshold until cost is better than target
threshold = 0.005
cost = compute_cost(threshold)
while cost >= target_cost and threshold <= 1.0:
	threshold += 0.005
	cost = compute_cost(threshold)
	print(threshold, cost)
if cost < target_cost:
	print('iter64_64_{}'.format(threshold))
