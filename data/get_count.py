import json
import math
import sys

fname = sys.argv[1]

with open(fname) as f:
	detections = json.load(f)

track_dict = {}
for frame_idx in range(len(detections)):
	if not detections[frame_idx]:
		continue
	for detection in detections[frame_idx]:
		detection['frame_idx'] = frame_idx
		track_id = detection['track_id']
		if track_id not in track_dict:
			track_dict[track_id] = []
		track_dict[track_id].append(detection)
tracks = track_dict.values()

def get_center(d):
	return (d['left']+d['right'])//2, (d['top']+d['bottom'])//2

def get_distance(d1, d2):
	x1, y1 = get_center(d1)
	x2, y2 = get_center(d2)
	dx = x1-x2
	dy = y1-y2
	return math.sqrt(dx*dx+dy*dy)

#print(len(tracks))

good_tracks = []
for track in tracks:
	duration = track[-1]['frame_idx'] - track[0]['frame_idx']
	distance = get_distance(track[0], track[-1])
	if duration < 10 or distance < 50:
		#print('skip', duration, distance)
		continue
	good_tracks.append(track)

print(len(good_tracks))
