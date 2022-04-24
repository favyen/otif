import os
import random
import subprocess

video_dir = '/data2/youtube/ytstream-dataset/data/uav/videos/'
video_fnames = {
    # fname to how many minutes
    '0006.mp4': 19,
    '0007.mp4': 19,
    '0008.mp4': 20,
    '0009.mp4': 21,
    #'0011.mp4': 20,
}
out_path = '/data2/blazeit/multiscope-test-set/uav/train/video/'

counter = 0
for fname, minutes in video_fnames.items():
    for min in range(minutes):
        ffmpeg_args = ['ffmpeg', '-ss', str(60*min), '-i', video_dir+fname, '-t', '60', out_path+str(counter)+'.mp4']
        subprocess.call(ffmpeg_args)
        counter += 1
