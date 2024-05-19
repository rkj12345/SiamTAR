from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob
import numpy as np
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR_3080_checkpot_e19 demo')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='../snapshot_r50/checkpoint_5data_EMA_xcorr_+_e19.pth', help='model name')
parser.add_argument('--update_path', default='../updatenet/best_checkpoint/checkpoint31.pth.tar', type=str, help='eval one special video')
parser.add_argument('--video_name', default='E:\\RGB_tracking_dataset\\OTB100\\uav1_bupt', type=str, help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    update_path = args.update_path
    step = 3
    tracker = SiamCARTracker(model, cfg.TRACK, step, update_path)

    hp = {'lr': 0.35, 'penalty_k': 0.2, 'window_lr': 0.45}

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    pred_bboxes = []
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
            # bbox = list(map(int, [213,121,21,95]))
            # cv2.rectangle(frame, (bbox[0], bbox[1]),
            #               (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            #               (0, 255, 0), 3)
            # cv2.imshow(video_name, frame)
            # cv2.waitKey(400000)
        else:
            state = tracker.track(frame, hp)
            pos = state['target_pos']  # cx, cy
            sz = state['target_sz']  # w, h
            # bbox = np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])
            # bbox = list(map(int, outputs['bbox']))
            bbox = list(map(int,[pos[0]-sz[0]/2 ,pos[1]-sz[1]/2, sz[0],sz[1]]))
            pred_bboxes.append(bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (255, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(1)
    model_path = os.path.join('../results_uav', 'bupt_uav')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, '{}.txt'.format('uav1_bupt'))
    with open(result_path, 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x]) + '\n')


if __name__ == '__main__':
    main()
