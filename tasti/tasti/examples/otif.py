import os
import cv2
import swag
import json
import tasti
import torch
import pandas as pd
import numpy as np
import torchvision
from scipy.spatial import distance
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler

import sys

data_root = sys.argv[1]
label = sys.argv[2]
out_fname = sys.argv[3]
ROOT_DATA_DIR = os.path.join(data_root, 'dataset', label)

with open(os.path.join(ROOT_DATA_DIR, 'cfg.json'), 'r') as f:
    cfg = json.load(f)
    gap = cfg['FPS']*5

best_detector = [fname for fname in os.listdir(os.path.join(ROOT_DATA_DIR, 'tracker')) if fname.startswith('yolov3-')][0]

'''
VideoDataset allows you to access frames of a given video.
'''
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_fp, transform_fn=lambda x: x):
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        self.cap = swag.VideoCapture(self.video_fp)
        self.video_metadata = json.load(open(self.video_fp + '.json', 'r'))
        self.cum_frames = np.array(self.video_metadata['cum_frames'])
        self.cum_frames = np.insert(self.cum_frames, 0, 0)
        self.length = self.cum_frames[-1]
        self.current_idx = 0
        self.init()

    def init(self):
        if len(self.list_of_idxs) == 0:
            self.frames = None
        else:
            self.frames = []
            for idx in tqdm(self.list_of_idxs, desc="Video"):
                self.seek(idx)
                frame = self.read()
                self.frames.append(frame)

    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform_fn(frame)
        return frame

    def seek(self, idx):
        if self.current_idx != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            self.current_idx = idx

    def read(self):
        _, frame = self.cap.read()
        frame = self.transform(frame)
        self.current_idx += 1
        return frame

    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)

    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            self.seek(idx)
            frame = self.read()
        else:
            frame = self.frames[idx]
        return frame

class Box:
    def __init__(self, box, object_name, confidence):
        self.box = box
        self.xmin = box[0]
        self.ymin = box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.object_name = object_name
        self.confidence = confidence

    def __str__(self):
        return f'Box({self.xmin},{self.ymin},{self.xmax},{self.ymax},{self.object_name},{self.confidence})'

    def __repr__(self):
        return self.__str__()

'''
LabelDataset loads the target dnn .csv files and allows you to access the target dnn outputs of given frames.
'''
class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, labels_fp, length):
        fnames = os.listdir(labels_fp)
        fnames = [fname for fname in fnames if fname.endswith('.json')]
        fnames.sort(key=lambda x: int(x.split('.')[0]))
        labels = []
        for fname in fnames:
            with open(os.path.join(labels_fp, fname), 'r') as f:
                detections = json.load(f)
            for dlist in detections:
                cur = []
                if dlist is None:
                    dlist = []
                for d in dlist:
                    cur.append(Box([d['left'], d['top'], d['right'], d['bottom']], d['class'], d['score']))
                labels.append(cur)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]

'''
Preprocessing function of a frame before it is passed to the Embedding DNN.
'''
def embedding_dnn_transform_fn(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

'''
Defines our notion of 'closeness' as described in the paper for two labels for only one object type.
'''
def is_close_helper(label1, label2):
    if len(label1) != len(label2):
        return False
    counter = 0
    for obj1 in label1:
        xavg1 = (obj1.xmin + obj1.xmax) / 2.0
        yavg1 = (obj1.ymin + obj1.ymax) / 2.0
        coord1 = [xavg1, yavg1]
        expected_counter = counter + 1
        for obj2 in label2:
            xavg2 = (obj2.xmin + obj2.xmax) / 2.0
            yavg2 = (obj2.ymin + obj2.ymax) / 2.0
            coord2 = [xavg2, yavg2]
            if distance.euclidean(coord1, coord2) < 100:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter

class OfflineIndex(tasti.Index):
    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model

    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model

    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model

    def get_target_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, 'train/video')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, 'tracker/video')
        video = VideoDataset(
            video_fp=video_fp
        )
        return video

    def get_embedding_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, 'train/video')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, 'tracker/video')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=embedding_dnn_transform_fn
        )
        return video

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        if train_or_test == 'train':
            labels_fp = os.path.join(ROOT_DATA_DIR, 'train', best_detector)
        else:
            labels_fp = os.path.join(ROOT_DATA_DIR, 'tracker', best_detector)
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache)
        )
        return labels

    def is_close(self, label1, label2):
        objects = set()
        for obj in (label1 + label2):
            objects.add(obj.object_name)
        for current_obj in list(objects):
            label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
            label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
            is_redundant = is_close_helper(label1_disjoint, label2_disjoint)
            if not is_redundant:
                return False
        return True

class LimitQuery(tasti.LimitQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)

class OfflineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True

        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000

if __name__ == '__main__':
    config = OfflineConfig()
    index = OfflineIndex(config)
    index.init()

    query = LimitQuery(index)
    res = query.execute_metrics(want_to_find=0, nb_to_find=10000, GAP=gap)

    # convert global frame index to (video_id, frame_idx)
    labels_fp = os.path.join(ROOT_DATA_DIR, 'tracker', best_detector)
    fnames = os.listdir(labels_fp)
    fnames = [fname for fname in fnames if fname.endswith('.json')]
    fnames.sort(key=lambda x: int(x.split('.')[0]))
    global_frame_to_local = []
    for fname in fnames:
        with open(os.path.join(labels_fp, fname), 'r') as f:
            nframes = len(json.load(f))
        for i in range(nframes):
            video_id = int(fname.split('.json')[0])
            global_frame_to_local.append((video_id, i))

    sel_frames = []
    for global_frame in res['ret_inds']:
        sel_frames.append(global_frame_to_local[global_frame])

    with open(out_fname, 'w') as f:
        json.dump(sel_frames, f)
