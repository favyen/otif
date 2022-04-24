# Extracts train/valid/test for amsterdam and jackson datasets.
# This is not needed unless preparing video from scratch.

import os
import random
import subprocess
import sys

data_root = sys.argv[1]
video_path = os.path.join(data_root, 'raw_video/blazeit/')

blazeit_length = 5
desired_length = 60
skip = desired_length//blazeit_length

for ds_id in ['amsterdam', 'jackson']:
    if ds_id == 'amsterdam':
        labels = ['amsterdam-2017-04-{}'.format(x) for x in ['10', '11', '12']]
    elif ds_id == 'jackson':
        labels = ['jackson-town-square-2017-12-{}'.format(x) for x in ['14', '16', '17']]
    else:
        raise Exception('unknown dataset ' + ds_id)

    for split in ['train', 'valid', 'test', 'tracker']:
        out_path = os.path.join(data_root, 'dataset', ds_id, split, 'video/')

        files = []
        for label in labels:
            for fname in os.listdir(os.path.join(video_path, label)):
                if not fname.endswith('.mp4'):
                    continue
                id = int(fname.split('.mp4')[0])
                if id % skip != 0:
                    continue
                files.append((label, id))

        files = random.sample(files, 60)
        counter = 0
        for label, id in files:
            fnames = [os.path.join(video_path, label, '{}.mp4'.format(id+i) for i in range(skip)]
            if not all([os.path.exists(fname) for fname in fnames]):
                print('skip {}/{} since some not exist'.format(label, id))
                continue
            ffmpeg_args = ['ffmpeg']
            for fname in fnames:
                ffmpeg_args.extend(['-i', fname])
            ffmpeg_args.append('-filter_complex')

            filter = ''
            for i in range(len(fnames)):
                filter += '[{}:v:0]'.format(i)
            filter += 'concat=n={}:v=1:a=0[outv]'.format(len(fnames))
            ffmpeg_args.append(filter)

            ffmpeg_args.extend(['-map', '[outv]', out_path+str(counter)+'.mp4'])
            subprocess.call(ffmpeg_args)
            counter += 1
