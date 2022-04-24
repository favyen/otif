import os
import subprocess
import sys

data_root = sys.argv[1]
label = sys.argv[2]
cls = sys.argv[3]

resolutions = (
	(640, 352),
	(416, 256),
	(320, 192),
	(224, 128),
	(160, 96),
)

for width, height in resolutions:
	data_path = os.path.join(data_root, 'dataset', label, 'train/seg-train/images/')
	model_path = os.path.join(data_root, 'dataset', label, 'segmentation_models', '{}_{}'.format(width, height), 'model')
	out_path = os.path.join(data_root, 'dataset', label, 'train/seg-train/', '{}_{}'.format(width, height))
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	os.makedirs(out_path, exist_ok=True)
	subprocess.call([
		'python', '../model/train.py',
		data_path, cls, str(width), str(height), model_path,
	])
	subprocess.call([
		'python', '../model/apply_train.py',
		data_path, str(width), str(height), model_path, out_path,
	])
