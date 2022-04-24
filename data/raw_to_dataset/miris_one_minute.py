# Extracts train/valid/test for tokyo and warsaw datasets.
# This is not needed unless preparing video from scratch.
# Note that the Tokyo dataset is called 'shibuya' everywhere in the code.

import os
import random
import subprocess
import sys

data_root = sys.argv[1]
video_path = os.path.join(data_root, 'raw_video/miris/')

for ds_name in ['shibuya', 'warsaw']:
    for video_idx, split in enumerate(['train', 'valid', 'test', 'tracker']):
        video_fname = os.path.join(video_path, ds_name, '{}.mp4'.format(video_idx))
        out_path = os.path.join(data_root, 'dataset', ds_name, split, 'video/')

        for i in range(50):
            ffmpeg_args = ['ffmpeg', '-ss', str(60*i), '-i', video_fname, '-t', '60', os.path.join(out_path, '{}.mp4'.format(i))]
            subprocess.call(ffmpeg_args)
