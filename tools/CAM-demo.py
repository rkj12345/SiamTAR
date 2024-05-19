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
import torchvision.utils as vutils
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker_cam import SiamCARTracker
from pysot.utils.model_load import load_pretrain
from CAM import get_CAM
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.utils.bbox import get_axis_aligned_bbox
import numpy as np
from toolkit.datasets import DatasetFactory
import torch.nn as nn
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR CAM demo')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--dataset_dir', type=str,default= 'E:\\RGB_tracking_dataset\\DTB70',
        help='datasets')
parser.add_argument('--dataset', type=str, default='DTB70',
        help='datasets')
parser.add_argument('--snapshot', type=str, default='../snapshot_r50/checkpoint_EMA_xcorr_+_e19.pth', help='model name')
parser.add_argument('--CAM_name', type=str, default='GradCAMpp', help='model name')
# softmax, head.cls_logits, head.centerness, head.bbox_tower.0, backbone.layer4.2, ....
parser.add_argument('--register_layer', default='softmax', type=str, help=' View the parameter names of each layer corresponding to the printed model')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
parser.add_argument('--format', default='bmp', type=str, help='png, pdf, jpg, bmp')
parser.add_argument('--save_dir', default='./test', type=str, help='Save path')
parser.add_argument('--update_path', default='../updatenet/models/checkpoint14.pth.tar', type=str, help='eval one special video')

args = parser.parse_args()

def show_cam(img, mask, title=None, title2=None):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        img (Tensor): shape (1, 3, H, W)
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # cam = heatmap + img.cpu()
    img = torch.from_numpy(img)
    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
    if title is not None:
        vutils.save_image(cam, title)
    if title is not None:
        vutils.save_image(heatmap, title2)

    return cam


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
        images = sorted(glob(os.path.join(video_name, '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame

class Model(nn.Module):
    def __init__(self, model):
        super(Model,self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def template(self, z):
        self.model.template(z)

    def template_return(self, z):
        zf = self.model.template_return(z)
        return zf

    def template_back(self, z):
        self.model.template_back(z)

    def track(self, x):
        outputs = self.model.track(x)
        outputs['cls'] = self.softmax(outputs['cls'])
        return outputs

def main():
    # load config
    cfg.merge_from_file(args.config)
    # hp_search
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    cur_dir = "E:\\RGB_tracking_dataset\\DTB70"
    dataset_root = os.path.join(cur_dir, args.dataset)
    update_path = args.update_path
    print(dataset_root)
    step = 3
    choose = True
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    print(model)
    model = Model(model)
    tracker = SiamCARTracker(model, cfg.TRACK, step, update_path)

    CAM = get_CAM(args.CAM_name, model=tracker, target_layer=args.register_layer)
    if args.dataset == "DTB70":
        dataset = "DTB70"
    else:
        dataset = args.dataset
    params = getattr(cfg.HP_SEARCH, dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    for v_idx, video in enumerate(dataset):
        if args.video_name != '':
            # test one special video
            if video.name != args.video_name:
                continue
        toc = 0
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                CAM.model.init(img, gt_bbox_)
            else:
                outputs, img_crop, bbox = CAM(img, hp)
                if outputs is not None:
                    heatmap_path = os.path.join(args.save_dir, args.CAM_name, "heatmap", video.name)
                    if os.path.exists(heatmap_path) is False:
                        os.makedirs(heatmap_path)
                    fusion_path = os.path.join(args.save_dir,args.CAM_name, "fusion", video.name)
                    if os.path.exists(fusion_path) is False:
                        os.makedirs(fusion_path)
                    show_cam(img_crop, outputs, fusion_path + "/{:06d}.{}".format(idx, args.format),
                             heatmap_path + "/{:06d}.{}".format(idx, args.format))
                    # scores.append(outputs['best_score'])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
        toc /= cv2.getTickFrequency()
        # save results
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))
if __name__ == '__main__':
    main()
