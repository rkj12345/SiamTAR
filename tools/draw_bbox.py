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
parser.add_argument('--snapshot', type=str, default='../snapshot_r50/checkpoint_5data_EMA_xcorr_+_e18.pth', help='model name')
parser.add_argument('--update_path', default='./updatenet/checkpoint/checkpoint3.pth.tar', type=str, help='eval one special video')
parser.add_argument('--video_name', default='E:\RGB_tracking_dataset\DTB70\DTB70\Animal3', type=str, help='videos or image files')
args = parser.parse_args()

def main():
    frame = cv2.imread(r'E:\\RGB_tracking_dataset\\DTB70\\DTB70\\Yacht4\\img\\00211.jpg')
    bbox = list(map(int, [633,188,52,32
]))
    cv2.rectangle(frame, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  (255, 255, 0), 1)
    cv2.imshow("111", frame)
    cv2.waitKey(40)

if __name__ == '__main__':
    main()