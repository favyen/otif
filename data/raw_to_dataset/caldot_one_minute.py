# Extracts train/valid/test for caldot1 and caldot2 datasets.
# This is not needed unless preparing video from scratch.

import os
import random
import subprocess
import sys

data_root = sys.argv[1]
video_path = os.path.join(data_root, 'raw_video/caldot/')

for ds_id in ['caldot1', 'caldot2']:
	out_paths = [
		os.path.join(data_root, 'dataset', ds_id, 'train/video/'),
		os.path.join(data_root, 'dataset', ds_id, 'valid/video/'),
		os.path.join(data_root, 'dataset', ds_id, 'test/video/'),
		os.path.join(data_root, 'dataset', ds_id, 'tracker/video/'),
	]

	def get_duration(fname):
		output = subprocess.check_output(['ffprobe', '-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'csv=s=,:p=0', fname])
		return float(output.strip())

	# list of tuples (fname, seconds)
	segments = []

	for fname in os.listdir(video_path):
		if not fname.startswith('{}-'.format(ds_id)):
			continue
		duration = int(get_duration(video_path+fname))
		for skip in range(0, duration-60, 60):
			segments.append((fname, skip))

	random.shuffle(segments)
	print('got {} segments'.format(len(segments)))

	for out_path in out_paths:
		cur_segments = segments[0:60]
		segments = segments[60:]
		for i, (fname, skip) in enumerate(cur_segments):
			ffmpeg_args = ['ffmpeg', '-ss', str(skip), '-i', video_path+fname, '-t', '60', out_path+str(i)+'.mp4']
			subprocess.call(ffmpeg_args)
