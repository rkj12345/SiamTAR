from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import cv2

from updatenet.net_upd import UpdateResNet512,UpdateResNet256
from .config import TrackerConfig
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg, step, update_path):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()
        self.state = dict()
        self.step = step  # 1,2,3
        if self.step == 1:
            self.name = 'SiamCAR_3080_checkpot_e19'
        elif self.step == 2:
            self.name = 'Linear'
        else:
            dataset = update_path.split('/')[-1].split('.')[0]
            if dataset == 'vot2018' or dataset == 'vot2016':
                self.name = 'UpdateNet'
            else:
                self.name = dataset
        if self.step == 3:
            # load UpdateNet network
            self.updatenet = UpdateResNet512()
            # self.updatenet = UpdateResNet256()
            update_model = torch.load(update_path)['state_dict']

            update_model_fix = dict()
            for i in update_model.keys():
                if i.split('.')[0] == 'module':  # 多GPU模型去掉开头的'module'
                    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
                else:
                    update_model_fix[i] = update_model[i]  # 单GPU模型直接赋值

            self.updatenet.load_state_dict(update_model_fix)

            # self.updatenet.load_state_dict(update_model)
            self.updatenet.eval().cuda()
        else:
            self.updatenet = ''

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        state = self.state
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        # target_pos = self.center_pos
        # target_sz = self.size
        #p = TrackerConfig

        # state['im_h'] = img.shape[0]
        # state['im_w'] = img.shape[1]
        # if p.adaptive:
        #     if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        #         p.instance_size = 287  # small object big search region
        #     else:
        #         p.instance_size = 255

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average) #[1, 3, 127, 127]
        z_crop = Variable(z_crop.unsqueeze(0))
        z_crop = z_crop.float()
        if self.step ==1:
            self.model.template(z_crop)
        else:
            z_f = self.model.template_return(z_crop.cuda())  # [1,512,6,6]
            z_f = torch.cat((z_f[0], z_f[1], z_f[2]), 1)  # [1, 768, 7, 7]
            self.model.template_back(z_f)
            state['z_f'] = z_f.cpu().data  # 累积的模板
            state['z_0'] = z_f.cpu().data  # 初始的模板
        # hanning = np.hanning(cfg.TRACK.SCORE_SIZE)  # (25,)
        # window = np.outer(hanning, hanning)  # (25, 25)

        #state['p'] = p
        # state['net'] = self.model
        # state['avg_chans'] = self.channel_average
        # state['window'] = window
        # state['target_pos'] = target_pos
        # state['target_sz'] = target_sz
        # self.state = state
        #
        # return state


    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, ltrbs, penalty_lk):

        # l+r [25, 25]  预测的是 ltrb=[0-l, 1-t, 2-r, 3-b]
        bboxes_w = ltrbs[0, :, :] + ltrbs[2, :, :]
        # t+b [25, 25]
        bboxes_h = ltrbs[1, :, :] + ltrbs[3, :, :]

        a = self.sz(bboxes_w, bboxes_h)

        b = self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)

        s_c = self.change(a / b)

        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))

        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)

        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        # print(max_r)
        # print(max_c)
        bbox_region = lrtbs[max_r, max_c, :]
        #print(bbox_region)
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        disp = self.accurate_location(max_r_up,max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy

    def track(self, img,hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        x_crop = Variable(x_crop.unsqueeze(0))
        x_crop = x_crop.float()

        outputs = self.model.track(x_crop)
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_score = penalty * cls * cen
        if cfg.TRACK.hanming:
            hp_score = p_score * (1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs, (1, 2, 0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr * ave_w + (1 - lr) * self.size[0]
        new_height = lr * ave_h + (1 - lr) * self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 0, img.shape[1])
        height = bbox_clip(new_height, 0, img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
        }