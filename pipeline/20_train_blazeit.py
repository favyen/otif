import os
import subprocess
import sys

data_root = sys.argv[1]
label = sys.argv[2]

width = 64
height = 64

data_path = os.path.join(data_root, 'dataset', label, 'train/blazeit-train/images/')
model_path = os.path.join(data_root, 'dataset', label, 'blazeit-model/model')
out_path = os.path.join(data_root, 'dataset', label, 'train/blazeit-train/out/')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(out_path, exist_ok=True)
subprocess.call([
	'python', '../blazeit/train.py',
	data_path, str(width), str(height), model_path,
])
subprocess.call([
	'python', '../blazeit/apply_train.py',
	data_path, str(width), str(height), model_path, out_path,
])
