import json
import numpy

def clip(x, lo, hi):
	if x < lo:
		return lo
	if x > hi:
		return hi
	return x


def load_target(fname, cls, input_dim, orig_dim, lenient=False):
	def input_clip(x, y):
		x = clip(x, 0, input_dim[0]-1)
		y = clip(y, 0, input_dim[1]-1)
		return int(x), int(y)

	with open(fname, 'r') as f:
		boxes = json.load(f)
	target = numpy.zeros((input_dim[1]//32, input_dim[0]//32), dtype='float32')
	for box in boxes:
		if box['class'] != cls:
			continue
		if lenient:
			padding = 0
		else:
			padding = max((box['bottom']-box['top'])//2, (box['right']-box['left'])//2)
		start = input_clip((box['left']-padding)*input_dim[0]/orig_dim[0], (box['top']-padding)*input_dim[1]/orig_dim[1])
		end = input_clip((box['right']+padding)*input_dim[0]/orig_dim[0], (box['bottom']+padding)*input_dim[1]/orig_dim[1])
		target[start[1]//32:end[1]//32+1, start[0]//32:end[0]//32+1] = 1
	return target

def update_rect(rect, x, y):
	if rect is None:
		return (x, y, x, y)
	return (
		min(x, rect[0]),
		min(y, rect[1]),
		max(x, rect[2]),
		max(y, rect[3]),
	)

class Component(object):
	def __init__(self):
		self.rect = None
		self.cells = []

	def add(self, x, y):
		self.rect = update_rect(self.rect, x, y)
		self.cells.append((x, y))

	def extend(self, component):
		self.rect = update_rect(self.rect, component.rect[0], component.rect[1])
		self.rect = update_rect(self.rect, component.rect[2], component.rect[3])
		self.cells.extend(component.cells)

	def __str__(self):
		return str(self.rect)

	def __repr__(self):
		return str(self)

# from binarized segmentation output, get a list of windows in which we should
# run the object detector.
# if sizes is set, we restrict windows to those sizes.
def get_windows(bin, sizes=None):
	bin = numpy.copy(bin)
	if bin.dtype != 'bool':
		raise Exception('expected bool not {}'.format(bin.dtype))

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
			component = Component()
			floodfill(component, x, y)
			components[counter] = component
			counter += 1

	# try to merge each component with its closest neighbor
	# only merge them if the needed-cell-to-area ratio increases
	def get_size(rect):
		rect_w, rect_h = rect[2]-rect[0]+1, rect[3]-rect[1]+1
		if sizes is None:
			return (rect_w, rect_h)
		# find smallest size in sizes that contains (w, h) window
		best = None
		for w, h in sizes:
			if w < rect_w or h < rect_h:
				continue
			if best is None or w*h < best[0]*best[1]:
				best = (w, h)
		return best

	initial_components = components.values()

	def get_ratio(component):
		w, h = get_size(component.rect)
		return len(component.cells)*1000//w//h

	check_ids = set(components.keys())
	while len(check_ids) > 0 and len(components) >= 2:
		id = check_ids.pop()
		component = components[id]
		orig_ratio = get_ratio(component)

		# find closest neighbor
		best_neighbor_id = None
		best_area = None
		for other_id, other in components.items():
			if other_id == id:
				continue
			rect = component.rect
			rect = update_rect(rect, other.rect[0], other.rect[1])
			rect = update_rect(rect, other.rect[2], other.rect[3])
			area = (rect[2]-rect[0]+1)*(rect[3]-rect[1]+1)
			if best_neighbor_id is None or area < best_area:
				best_neighbor_id = other_id
				best_area = area

		# start producing a merged component
		used_components = set([id, best_neighbor_id])
		merged_comp = Component()
		merged_comp.extend(component)
		merged_comp.extend(components[best_neighbor_id])
		w, h = get_size(merged_comp.rect)

		# try to fit in any other components that we can, without increasing the window
		for other_id, other in components.items():
			if other_id in used_components:
				continue
			rect = merged_comp.rect
			rect = update_rect(rect, other.rect[0], other.rect[1])
			rect = update_rect(rect, other.rect[2], other.rect[3])
			if rect[2]-rect[0]+1 > w or rect[3]-rect[1]+1 > h:
				continue

			# it fits!
			used_components.add(other_id)
			merged_comp.extend(other)

		# now do the ratio test
		new_ratio = get_ratio(merged_comp)
		if new_ratio < orig_ratio:
			continue

		for comp_id in used_components:
			del components[comp_id]
			if comp_id in check_ids:
				check_ids.remove(comp_id)
		components[counter] = merged_comp
		check_ids.add(counter)
		counter += 1

	return components.values()
