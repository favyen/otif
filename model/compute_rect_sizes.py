import json
import os
import skimage.io
import sys

import util

data_root = sys.argv[1]
label = sys.argv[2]

with open(os.path.join(data_root, 'dataset', label, 'cfg.json'), 'r') as f:
	cfg = json.load(f)
	cls_set = cfg['Classes'].keys()
	cls = list(cls_set)[0]
	orig_dims = cfg['OrigDims']

data_path = os.path.join(data_root, 'dataset', label, 'train/seg-train/images/')
examples = []
fnames = [fname for fname in os.listdir(data_path) if fname.endswith('.json')]
for i, fname in enumerate(fnames):
	target = util.load_target(os.path.join(data_path, fname), cls, orig_dims, orig_dims, lenient=True)
	examples.append(target)

# cluster cells in each image by density
# only consider non-empty images
ex_lists = []
for target in examples:
	bin = target > 0.5

	# use floodfill to get initial list of windows
	# rationale: we don't want any windows that cut through the middle of an object
	def floodfill(component, x, y):
		if x < 0 or x >= bin.shape[1] or y < 0 or y >= bin.shape[0]:
			return
		elif not bin[y, x]:
			return
		bin[y, x] = False
		component.add(x, y)
		for x_offset in [-1, 0, 1]:
			for y_offset in [-1, 0, 1]:
				floodfill(component, x+x_offset, y+y_offset)

	components = {}
	counter = 0
	for y in range(bin.shape[0]):
		for x in range(bin.shape[1]):
			if not bin[y, x]:
				continue
			component = util.Component()
			floodfill(component, x, y)
			components[counter] = component
			counter += 1

	# repeatedly find the two components that yield highest density when merged
	# keep track of the rectangles at each step
	rect_list = []
	def save_rects():
		rect_list.append([(c.rect[2]-c.rect[0], c.rect[3]-c.rect[1]) for c in components.values()])

	save_rects()
	while len(components) > 1:
		best = None
		best_density = None
		for i, comp1 in components.items():
			# attempt merge with closest neighbor
			best_neighbor_id = None
			best_area = None
			for j, comp2 in components.items():
				if i == j:
					continue
				rect = comp1.rect
				rect = util.update_rect(rect, comp2.rect[0], comp2.rect[1])
				rect = util.update_rect(rect, comp2.rect[2], comp2.rect[3])
				area = (rect[2]-rect[0]+1)*(rect[3]-rect[1]+1)
				if best_neighbor_id is None or area < best_area:
					best_neighbor_id = j
					best_area = area

			merged_comp = util.Component()
			merged_comp.extend(comp1)
			merged_comp.extend(components[best_neighbor_id])
			density = len(merged_comp.cells)*1000//best_area
			if best is None or density > best_density:
				best = (i, best_neighbor_id)
				best_density = density

		components[best[0]].extend(components[best[1]])
		del components[best[1]]
		save_rects()

	if len(rect_list) == 0:
		continue

	# save only the last three
	rect_list = rect_list[-3:]
	ex_lists.append(rect_list)

print('got {} examples (originally {})'.format(len(ex_lists), len(examples)))

scale = 32
big_rect = ((orig_dims[0]+scale-1)//scale, (orig_dims[1]+scale-1)//scale)
picked = [big_rect]

def check_time(det_list, rects):
	t = 0
	for rect in rects:
		best = None
		for det_r in det_list:
			if det_r[0] < rect[0] or det_r[1] < rect[1]:
				continue
			area = det_r[0]*det_r[1]
			if best is None or area < best:
				best = area
		if best is None:
			raise Exception('bad det_list')
		t += best
	return t

def evaluate(det_list):
	total_time = 0
	for rect_list in ex_lists:
		best_time = None
		for rects in rect_list:
			t = check_time(det_list, rects)
			if best_time is None or t < best_time:
				best_time = t
		total_time += best_time
	return total_time

print('orig_dims={} so start with {}'.format(orig_dims, picked))
while len(picked) < 3:
	best = None
	best_time = None
	for w in range(1, big_rect[0]):
		for h in range(1, big_rect[1]):
			cur_time = evaluate(picked + [(w, h)])
			if best is None or cur_time < best_time:
				best = (w, h)
				best_time = cur_time
	picked.append(best)
	print('update picked:', picked)

print([(w*32, h*32) for w, h in picked])
