# Given list of detections in the frame, score how likely it satisfies the query.
# Returns 0 if it almost certainly doesn't satisfy the query.
# Otherwise returns a positive real number, and providing other arguments can help with it.
# In addition to the score, also returns relevant detections and a separate count for training BlazeIt model.
def get_score(query_name, dlist, track_lengths=None):
	def get_min_displacement(dlist, n):
		cur_displacements = []
		for d in dlist:
			cur_displacements.append(track_lengths[d['track_id']]+1)
		cur_displacements.sort()
		return min(cur_displacements[-n:])

	def count_query_score(dlist, n):
		if len(dlist) < n:
			return 0, [], len(dlist)
		if track_lengths is not None:
			return get_min_displacement(dlist, n), dlist, len(dlist)
		else:
			return len(dlist), dlist, len(dlist)

	def spatial_query_score(dlist, radius, n):
		# O(n^2) algorithm: for each detection, find nearby detections, then take best score.
		best_score = 0
		best_dlist = []
		train_count = 0 # binary either 0 or 1
		for src in dlist:
			src_x = (src['left']+src['right'])//2
			src_y = (src['top']+src['bottom'])//2
			# Find neighbors.
			neighbors = []
			for cur in dlist:
				cur_x = (cur['left']+cur['right'])//2
				cur_y = (cur['top']+cur['bottom'])//2
				d = (cur_x-src_x)**2 + (cur_y-src_y)**2
				if d > radius*radius:
					continue
				neighbors.append(cur)
			if len(neighbors) < n:
				continue
			train_count = 1
			if track_lengths is not None:
				score = get_min_displacement(neighbors, n)
			else:
				score = len(neighbors)
			if score > best_score:
				best_score = score
				best_dlist = neighbors

		return best_score, best_dlist, train_count

	# Detection count queries.
	if query_name == 'uav':
		dlist = [d for d in dlist if d['class'] == 'car']
		return count_query_score(dlist, 32)
	elif query_name == 'shibuya':
		dlist = [d for d in dlist if d['class'] == 'car']
		return count_query_score(dlist, 17)
	elif query_name == 'taipei':
		dlist = [d for d in dlist if d['class'] == 'car']
		return count_query_score(dlist, 6)

	# Region-based queries.
	elif query_name == 'jackson':
		dlist = [d for d in dlist if (d['top']+d['bottom'])//2 > 400 and d['class'] == 'car']
		return count_query_score(dlist, 4)
	elif query_name == 'caldot1':
		dlist = [d for d in dlist if (d['top']+d['bottom'])//2 > 230 and d['class'] == 'car']
		return count_query_score(dlist, 11)

	# Spatial relationships.
	elif query_name == 'warsaw':
		dlist = [d for d in dlist if (d['top']+d['bottom'])//2 > 275 and d['class'] == 'car']
		return spatial_query_score(dlist, 50, 4)
	elif query_name == 'amsterdam':
		dlist = [d for d in dlist if d['class'] == 'car']
		return spatial_query_score(dlist, 200, 2)

	raise Exception('unknown query {}'.format(query_name))
