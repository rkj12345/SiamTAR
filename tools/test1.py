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
#这个地如果加重定位的话，需要改动引入的tracker类
from pysot.tracker.siamcar_tracker_ori import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
# from my_eval import evaluation
from toolkit.datasets import DatasetFactory
# from toolkit.utils.region import vot_overlap, vot_float2str

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='GOT10K',
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=True,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='../snapshot_r50/checkpoint_e19.pth',
        help='snapshot_oral of models to eval')
parser.add_argument('--update_path', default='../updatenet/best_checkpoint/checkpoint31.pth.tar', type=str, help='eval one special video')

parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)
    # hp_search
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    cur_dir = "E:\\BaiduNetdiskDownload"
    dataset_root = os.path.join(cur_dir,args.dataset)
    update_path = args.update_path
    print(dataset_root)
    step = 3
    choose = True
    model = ModelBuilder()
    snapshot = args.snapshot


    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK,  step, update_path)

    # create dataset
    dataset_root = 'G:\\21-23-tracking_dataset\\RGB\\GOT10K\\test'
    #dataset_root = '/autodl-tmp/VOT2018'
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = tracker.name

    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        track_times = []

        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()  # 用于计时，初始的计时周期
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                state = tracker.track(img, hp)

                pred_bbox = state['bbox']  # w, h
                pred_bbox = np.array(pred_bbox)
                # pred_bbox = np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if not any(map(math.isnan, gt_bbox)):
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))

                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (255, 255, 0), 3)

                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (255, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
        model_path = os.path.join('D:\\codeProjects\\SiamCAR-master-updateneting\\results', args.dataset, model_name, video.name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        result_path = os.path.join(model_path, '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        result_path = os.path.join(model_path,
                                   '{}_time.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in track_times:
                f.write("{:.6f}\n".format(x))
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))
    os.chdir(model_path)
    save_file = '../%s' % dataset
    shutil.make_archive(save_file, 'zip')
    print('Records saved at', save_file + '.zip')

    #evaluation(args.dataset, model_name)

if __name__ == '__main__':
    main()