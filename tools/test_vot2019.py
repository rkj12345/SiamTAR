# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')
from tqdm import tqdm
from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
# from my_eval import evaluation
from toolkit.datasets import DatasetFactory

# from toolkit.utils.region import vot_overlap, vot_float2str


from utils import poly_iou

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='VOT2019',
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=False,
        help='whether visualzie result')
parser.add_argument('-j', '--workers', default=11, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--snapshot', type=str, default='../snapshot_r50/checkpoint_5data_EMA_xcorr_+_e18.pth',
        help='snapshot_oral of models to eval')
parser.add_argument('--update_path', default='./updatenet/checkpoint/checkpoint1.pth.tar', type=str, help='eval one special video')

parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)
    # hp_search
    params = getattr(cfg.HP_SEARCH, args.dataset)
    print(1111)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    cur_dir = "E:\\RGB_tracking_dataset"
    dataset_root = os.path.join(cur_dir,args.dataset)
    update_path = args.update_path
    print(dataset_root)
    step = 1
    choose = True
    model = ModelBuilder()


    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK,  step, update_path)

    # create dataset
    dataset_root = 'E:\\RGB_tracking_dataset\\VOT2019'
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = tracker.name
    #OPE tracking
    total_lost = 0
    for video in tqdm(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        frame_counter = 0
        lost_number = 0
        pred_bboxes = []
        track_times = []
        state= dict()
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
            tic = cv2.getTickCount() #用于计时，初始的计时周期
            if idx == frame_counter:

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                state = tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(1)
            elif idx > frame_counter:
                #目标框的左上角坐标，以及宽高
                state = tracker.track(img, hp)
                pos = state['target_pos']  # cx, cy
                sz = state['target_sz']  # w, h
                pred_bbox = np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])

                #pred_bboxes.append(pred_bbox)
                #pred_bboxes.append(pred_bbox)
                overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))#ddadasd
                if overlap>0:
                    pred_bboxes.append(pred_bbox)
                else:
                    #lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5  # skip 5 frames
                    lost_number += 1
            else:
                pred_bboxes.append(0)

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                #标记GT黄色框信息
                cv2.polylines(img, [np.array(gt_bbox, np.int_).reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)

                if cfg.MASK:
                    cv2.polylines(img, [np.array(pred_bbox, np.int_).reshape((-1, 1, 2))],
                                  True, (0, 255, 255), 3)
                else:
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
        model_path = os.path.join('../results', args.dataset, model_name, 'baseline', video.name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        result_path = os.path.join(model_path, '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                if isinstance(x, int):
                    f.write("{:d}\n".format(x))
                else:
                    f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
        total_lost += lost_number
        ##########################################################################
        # result_path = os.path.join(model_path,
        #             '{}_time.txt'.format(video.name))
        # with open(result_path, 'w') as f:
        #     for x in track_times:
        #         f.write("{:.6f}\n".format(x))
        # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        #     v_idx+1, video.name, toc, idx / toc))
    # os.chdir(model_path)
    # save_file = '../%s' % dataset
    # shutil.make_archive(save_file, 'zip')
    # print('Records saved at', save_file + '.zip')
    evaluation(args.dataset, model_name)

if __name__ == '__main__':
    main()