# Copyright (c) SenseTime. All Rights Reserved.

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

from siamfc_backbone.heads import SiamFC
from siamfc_backbone.backbones import  AlexNetV1
from re_location.re_locate import relocate
from re_location.detect import  detect
from re_location.Re_locate2 import relocate as relocate2
import skimage.feature
import skimage.segmentation
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

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
        self.is_deterministic = True
        self.device = torch.device("cuda")
        '''********************************************************************************************************'''
        self.reloca_model = Net(
            backbone=AlexNetV1(),
            head=SiamFC(0.001))
        self.reloca_model.load_state_dict(
            torch.load(
                "D:\\codeProjects\\zhangruixing\\SiamCAR_3080_checkpot_e19\\SiamCAR_TRAM\\SiamCAR_alex\\re_location\\pth\\siamfc_alexnet_e50.pth",
                map_location=lambda storage, loc: storage))
        cuda = torch.cuda.is_available()
        self.device_fc = torch.device('cuda:0' if cuda else 'cpu')
        self.reloca_model = self.reloca_model.to(self.device_fc)
        self.reloca_model.eval()
        '''*********************************************************************************************************'''

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def init(self, img, bbox, img_file):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        '''# #*******************************************************************************************************'''
        self.use_relocate = True
        box = bbox

        if self.use_relocate:
            self.Occlusion = 0

            # 用于记录各种权重
            self.color_weight = 0.6
            self.deep_feature_weight = 0.4

            # 刚性物体检测需要的权重
            self.color_weight1 = 0.2
            self.deep_feature_weight1 = 0.4
            self.texture_weight = 0.4 * 0.5

            self.first_weight = 0.9
            self.recent_weight = 0.1

            self.distance_weight = 0.7

            self.first_flash_weight = 0.55

            # 目标检测尺度与target_sz
            self.scale = 0.9

            # 特征更新所跨帧数
            self.flash_feature = 5

            # 记录一次重定位之后的有效范围
            self.flag_record = 0
            self.flag_record_original = 10

            # 更新阈值所跨的帧数
            self.flash_num = 25
            self.flag_diff = 25

            # 开始使用重定位的起点
            self.start_flash = 25

            # 用于记录diff的sum和max，用于阈值更新
            self.diff_sum = 0
            self.diff_max = 0

            # 记录目标类别
            self.template_class = -1024

            # 用于记录每一帧产生的diff
            self.color_diff = 0
            self.deep_feature_diff = 0
            self.diff = 0

            # 截取目标块（原数据）
            w = int(box[2])
            x = int(box[0])
            h = int(box[3])
            y = int(box[1])
            img_template = img[y: y + h, x:x + w, :]
            img_template = cv2.resize(img_template, (w, h), interpolation=cv2.INTER_CUBIC)

            # 检测类别
            template_class, template_h, template_w, template_y, template_x = detect(img_file, box)
            self.template_class = template_class

            #判别阈值
            if self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22:
                self.diff_threshlod = 0.415
                if "Human3" in img_file or "Girl2" in img_file or "Basketball" in img_file or "Walking2" in img_file or "girl" in img_file:
                    if "Basketball" in img_file or "girl" in img_file:
                        self.use_relocate = False
                    pass
                else:
                    self.use_relocate = False
            else:
                self.use_relocate = False  #原来是false
                self.diff_threshlod = 0.2

            if self.template_class == -1:
                self.use_relocate = False
            else:
                img_template_detect = img[int(template_y):int(template_y + template_h),
                                      int(template_x):int(template_x + template_w), :]
                img_template_detect1 = cv2.resize(img_template_detect, (96, 96), interpolation=cv2.INTER_CUBIC)

                ##提取深度特征(目标检测）
                img_template_detect1 = torch.from_numpy(img_template_detect1).to(
                    self.device_fc).permute(2, 0, 1).unsqueeze(0).float()
                self.template_deep_feature_detect = torch.squeeze(
                    self.reloca_model.backbone(img_template_detect1)).view(256, 4)

                # 获取目标的颜色信息(目标检测）
                template_center_x = 0.5 * w
                template_center_y = 0.5 * h
                ratio = 0.3
                local_color = img_template_detect[
                              int(template_center_y - ratio * h):int(template_center_y + ratio * h),
                              int(template_center_x - ratio * w):int(template_center_x + ratio * w)]
                self.template_color_detect = torch.tensor((cv2.calcHist([local_color], [0], None, [256], [0.0, 255.0]),
                                                           cv2.calcHist([local_color], [1], None, [256], [0.0, 255.0]),
                                                           cv2.calcHist([local_color], [2], None, [256],
                                                                        [0.0, 255.0]))).squeeze()

            if self.use_relocate:
                self.template_box = box

                # 存储目标的长宽(GT)
                self.template_w_original = box[2]
                self.template_h_original = box[3]

                ##提取深度特征(GT)
                img_template1 = cv2.resize(img_template, (96, 96), interpolation=cv2.INTER_CUBIC)
                img_template1 = torch.from_numpy(img_template1).to(
                    self.device_fc).permute(2, 0, 1).unsqueeze(0).float()
                self.template_deep_feature = torch.squeeze(self.reloca_model.backbone(img_template1)).view(256, 4)

                # 记录纹理特征
                img_template_texture = img_template.copy()
                patch_center_x = int(0.5 * w)
                patch_center_y = int(0.5 * h)

                train_hist = [[], [], []]  # ,[], [], []]

                img_template_texture1 = img_template_texture[
                                        int(patch_center_y - 0.5 * h):int(patch_center_y),
                                        int(patch_center_x - 0.5 * w):int(patch_center_x)]
                img_template_texture2 = img_template_texture[
                                        int(patch_center_y - 0.5 * h):int(patch_center_y),
                                        int(patch_center_x):int(patch_center_x + 0.5 * w)]
                img_template_texture3 = img_template_texture[
                                        int(patch_center_y):int(patch_center_y + 0.5 * h),
                                        int(patch_center_x - 0.5 * w):int(patch_center_x)]
                img_template_texture4 = img_template_texture[
                                        int(patch_center_y):int(patch_center_y + 0.5 * h),
                                        int(patch_center_x):int(patch_center_x + 0.5 * w)]
                for colour_channel in (0, 1, 2):
                    img_template_texture1[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_template_texture1[:, :, colour_channel], 8, 1.0, method='var')
                    img_template_texture2[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_template_texture2[:, :, colour_channel], 8, 1.0, method='var')
                    img_template_texture3[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_template_texture3[:, :, colour_channel], 8, 1.0, method='var')
                    img_template_texture4[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_template_texture4[:, :, colour_channel], 8, 1.0, method='var')

                    img_template_texture01 = img_template_texture1[:, :, colour_channel] - img_template_texture2[:,
                                                                                           :, colour_channel]
                    img_template_texture02 = img_template_texture3[:, :, colour_channel] - img_template_texture4[:,
                                                                                           :, colour_channel]
                    img_template_texture00 = img_template_texture01 - img_template_texture02

                    # 统计直方图
                    train_hist[colour_channel], _ = np.histogram(img_template_texture00,
                                                                 density=False,
                                                                 bins=256, range=(0, 256))
                    # train_hist[colour_channel+3], _ = np.histogram(img_template_texture02,
                    #                                              density=False,
                    #                                              bins=256, range=(0, 256))
                self.texture = torch.tensor(train_hist).squeeze()

                # 获取目标的颜色信息
                template_center_x = 0.5 * w
                template_center_y = 0.5 * h
                ratio = 0.3
                local_color = img_template[
                              int(template_center_y - ratio * h):int(template_center_y + ratio * h),
                              int(template_center_x - ratio * w):int(template_center_x + ratio * w)]
                self.template_color = torch.tensor((cv2.calcHist([local_color], [0], None, [256], [0.0, 255.0]),
                                                    cv2.calcHist([local_color], [1], None, [256], [0.0, 255.0]),
                                                    cv2.calcHist([local_color], [2], None, [256],
                                                                 [0.0, 255.0]))).squeeze()

                self.template_deep_feature_first = self.template_deep_feature_detect
                self.template_color_first = self.template_color_detect
                self.template_texture_first = self.texture
        '''# # ******************************************************************************************************'''
        state = self.state
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        target_pos = self.center_pos
        target_sz = self.size
        p = TrackerConfig

        state['im_h'] = img.shape[0]
        state['im_w'] = img.shape[1]
        if p.adaptive:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = 287  # small object big search region
            else:
                p.instance_size = 255

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
        hanning = np.hanning(cfg.TRACK.SCORE_SIZE)  # (25,)
        window = np.outer(hanning, hanning)  # (25, 25)

        state['p'] = p
        state['net'] = self.model
        state['avg_chans'] = self.channel_average
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        self.state = state

        return state


    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
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
        bbox_region = lrtbs[max_r, max_c, :]
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

    def track(self, img, hp, idx, img_file):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        state = self.state
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        w_z = target_sz[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(target_sz)
        h_z = target_sz[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(target_sz)
        s_z = np.sqrt(w_z * h_z)

        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average) #[3, 255, 255]
        #outputs['cls']:[1, 2, 25, 25]  outputs['loc']:[1, 4, 25, 25]  outputs['cen']:[1, 1, 25, 25]
        x_crop = Variable(x_crop.unsqueeze(0))
        x_crop = x_crop.float()
        #outputs['cls'].shape:[1, 2, 25, 25] cen:[1, 1, 25, 25]  loc:[1, 4, 25, 25]
        outputs = self.model.track(x_crop)#输出为三分支特征图，

        cls = self._convert_cls(outputs['cls']).squeeze()#(25, 25)
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()#(25, 25)
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()#(4, 25, 25)

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1#193
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])#[25,25]
        p_score = penalty * cls * cen#(25, 25)
        if cfg.TRACK.hanming:#汉宁窗惩罚，
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC) #(193, 193)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC) #(193, 193)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)# (193, 193)
        lrtbs = np.transpose(lrtbs,(1,2,0)) # (25, 25, 4)
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)#(193, 193, 4)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k']) # 0.8779869035415468
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr*ave_w + (1-lr)*self.size[0]
        new_height = lr*ave_h + (1-lr)*self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx,0,img.shape[1])
        cy = bbox_clip(new_cy,0,img.shape[0])
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        #目标框的左上角坐标，以及宽高
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        '''#*************************************++++++++++++++++++++++++++++++++++++++++++++++++********************'''
        if self.use_relocate:
            # center = target_pos
            target_sz = self.size
            center = self.center_pos
            temp_x1 = int(center[0] + 1 - (target_sz[0] - 1) / 2)
            if temp_x1 < 0:
                temp_x1 = 0
            temp_x2 = temp_x1 + int(target_sz[0])
            if temp_x2 > img.shape[1]:
                temp_x2 = img.shape[1]
            temp_y1 = int(center[1] + 1 - (target_sz[1] - 1) / 2)
            if temp_y1 < 0:
                temp_y1 = 0
            temp_y2 = temp_y1 + int(target_sz[1])
            if temp_y2 > img.shape[0]:
                temp_y2 = img.shape[0]

            patch_img = img[temp_y1:temp_y2, temp_x1:temp_x2, :]

            w = int(self.template_w_original)
            h = int(self.template_h_original)
            patch_img = cv2.resize(patch_img, (w, h), interpolation=cv2.INTER_CUBIC)

            # 提取纹理特征
            if not (
                    self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22):
                img_patch_texture = patch_img.copy()
                patch_center_x = int(0.5 * w)
                patch_center_y = int(0.5 * h)

                train_hist = [[], [], [], [], [], []]

                img_patch_texture1 = img_patch_texture[
                                     int(patch_center_y - 0.5 * h):int(patch_center_y),
                                     int(patch_center_x - 0.5 * w):int(patch_center_x)]
                img_patch_texture2 = img_patch_texture[
                                     int(patch_center_y - 0.5 * h):int(patch_center_y),
                                     int(patch_center_x):int(patch_center_x + 0.5 * w)]
                img_patch_texture3 = img_patch_texture[
                                     int(patch_center_y):int(patch_center_y + 0.5 * h),
                                     int(patch_center_x - 0.5 * w):int(patch_center_x)]
                img_patch_texture4 = img_patch_texture[
                                     int(patch_center_y):int(patch_center_y + 0.5 * h),
                                     int(patch_center_x):int(patch_center_x + 0.5 * w)]

                for colour_channel in (0, 1, 2):
                    img_patch_texture1[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_patch_texture1[:, :, colour_channel], 8, 1.0, method='var')
                    img_patch_texture2[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_patch_texture2[:, :, colour_channel], 8, 1.0, method='var')
                    img_patch_texture3[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_patch_texture3[:, :, colour_channel], 8, 1.0, method='var')
                    img_patch_texture4[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                        img_patch_texture4[:, :, colour_channel], 8, 1.0, method='var')

                    img_template_texture01 = img_patch_texture1[:, :, colour_channel] - img_patch_texture2[:, :,
                                                                                        colour_channel]
                    img_template_texture02 = img_patch_texture2[:, :, colour_channel] - img_patch_texture4[:, :,
                                                                                        colour_channel]
                    # img_template_texture00 = img_template_texture01 - img_template_texture02

                    # 统计直方图
                    train_hist[colour_channel], _ = np.histogram(img_template_texture01,
                                                                 density=False,
                                                                 bins=256, range=(0, 256))
                    train_hist[colour_channel + 3], _ = np.histogram(img_template_texture02,
                                                                     density=False,
                                                                     bins=256, range=(0, 256))

                patch_texture = torch.tensor(train_hist).squeeze()

                texture_diff = int(
                    sum(torch.nn.functional.pairwise_distance(self.texture, patch_texture, p=2, eps=1e-06)))

            ##提取深度特征
            patch_img1 = cv2.resize(patch_img, (96, 96), interpolation=cv2.INTER_CUBIC)
            patch_img1 = torch.from_numpy(patch_img1).to(
                self.device).permute(2, 0, 1).unsqueeze(0).float()
            patch_deep_feature = torch.squeeze(self.reloca_model.backbone(patch_img1)).view(256, 4)

            if self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22:
                deep_feature_diff = int(
                    sum(torch.nn.functional.pairwise_distance(self.template_deep_feature, patch_deep_feature,
                                                              p=2, eps=1e-06)))
            else:
                deep_feature_diff = int(
                    sum(torch.nn.functional.pairwise_distance(self.template_deep_feature_detect, patch_deep_feature,
                                                              p=2, eps=1e-06)))

            ##颜色特征提取
            patch_center_x = 0.5 * w
            patch_center_y = 0.5 * h
            ratio = 0.3
            local_color = patch_img[int(patch_center_y - ratio * h):int(patch_center_y + ratio * h),
                          int(patch_center_x - ratio * w):int(patch_center_x + ratio * w)]

            patch_color = torch.tensor((cv2.calcHist([local_color], [0], None, [256], [0.0, 255.0]),
                                        cv2.calcHist([local_color], [1], None, [256], [0.0, 255.0]),
                                        cv2.calcHist([local_color], [2], None, [256], [0.0, 255.0]))).squeeze()
            color_diff = int(
                sum(torch.nn.functional.pairwise_distance(self.template_color, patch_color, p=2, eps=1e-06)))

            color_weight = self.color_weight
            deep_feature_weight = self.deep_feature_weight
            first_weight = self.first_weight
            recent_weight = self.recent_weight
            distance_weight = self.distance_weight

            if self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22:
                diff = int(color_diff * color_weight + deep_feature_diff * deep_feature_weight)
            else:
                diff = int(
                    texture_diff * self.texture_weight + color_diff * self.color_weight1 + deep_feature_diff * self.deep_feature_weight1)

            if self.flag_record > 0 and diff < self.diff + self.diff * (self.diff_threshlod * 2):
                self.flag_record = self.flag_record - 1
                if not (
                        self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22):
                    pass
                    # print(
                    #     "重定位后，第{}帧跟踪结果,color_diff:{}   texture_diff：{}    deep_diff:{}    diff:{}    阈值:{}".format(idx,
                    #                                                                                                color_diff,
                    #                                                                                                texture_diff,
                    #                                                                                                deep_feature_diff,
                    #                                                                                                diff,
                    #                                                                                                self.diff))
                else:
                    pass
                    # print("重定位后，第{}帧跟踪结果,color_diff:{}   deep_diff:{}    diff:{}    阈值：{}".format(idx,
                    #                                                                               color_diff,
                    #                                                                               deep_feature_diff,
                    #                                                                               diff,
                    #                                                                               self.diff))
            elif idx <= self.start_flash:
                if not (
                        self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22):
                    pass
                    # print(
                    #     "第{}帧跟踪结果,color_diff:{}   texture_diff：{}    deep_diff:{}    diff:{}    阈值:{}".format(idx,
                    #                                                                                           color_diff,
                    #                                                                                           texture_diff,
                    #                                                                                           deep_feature_diff,
                    #                                                                                           diff,
                    #                                                                                           self.diff))
                else:
                    pass
                    # print("第{}帧跟踪结果,color_diff:{}   deep_diff:{}    diff:{}    阈值：{}".format(idx,
                    #                                                                          color_diff,
                    #                                                                          deep_feature_diff,
                    #                                                                          diff,
                    #                                                                          self.diff))

            elif diff < self.diff + self.diff * self.diff_threshlod:

                if not (
                        self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22):
                    pass
                    # print("第{}帧跟踪成功,color_diff:{}   texture_diff:{}    deep_diff:{}    diff:{}    阈值：{}".format(idx,
                    #                                                                                             color_diff,
                    #                                                                                             texture_diff,
                    #                                                                                             deep_feature_diff,
                    #                                                                                             diff,
                    #                                                                                             self.diff))
                else:
                    pass
                    # print("第{}帧跟踪成功,color_diff:{}   deep_diff:{}    diff:{}    阈值：{}".format(idx,
                    #                                                                          color_diff,
                    #                                                                          deep_feature_diff,
                    #                                                                          diff,
                    #                                                                          self.diff))
            else:

                if not (
                        self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22):
                    pass
                    # print("第{}帧跟踪失败,color_diff:{}   texture_diff:{}    deep_diff:{}    diff:{}    阈值：{}".format(idx,
                    #                                                                                             color_diff,
                    #                                                                                             texture_diff,
                    #                                                                                             deep_feature_diff,
                    #                                                                                             diff,
                    #                                                                                             self.diff))
                else:
                    pass
                    # print("第{}帧跟踪失败,color_diff:{}   deep_diff:{}    diff:{}    阈值：{}".format(idx,
                    #                                                                          color_diff,
                    #                                                                          deep_feature_diff,
                    #                                                                          diff,
                    #                                                                          self.diff))

                if self.template_class == 0 or self.template_class == 15 or self.template_class == 17 or self.template_class == 22:
                    new_box, t_diff, color, deep = relocate(img_file, self.template_deep_feature,
                                                            self.template_color,
                                                            self.template_class, self.template_box, color_weight,
                                                            deep_feature_weight,
                                                            center[1], center[0],
                                                            target_sz[0] * target_sz[1], distance_weight,
                                                            target_sz[1], target_sz[0])

                    if t_diff != -1 and t_diff <= diff * 1.1:
                        diff = t_diff * 0.5 + self.diff * (1 + self.diff_threshlod * 2) * 0.5
                        x1 = new_box[0]
                        x2 = new_box[2]
                        y1 = new_box[1]
                        y2 = new_box[3]
                        center_y = y1 + (y2 - y1) / 2
                        center_x = x1 + (x2 - x1) / 2

                        self.center_pos = np.array([center_x, center_y])
                        if (x2 - x1) <= target_sz[0] * 1.2:
                            width = x2 - x1
                            height = y2 - y1
                        bbox = [center_x - width / 2,
                                center_y - height / 2,
                                width,
                                height]

                        patch_color = color
                        patch_deep_feature = deep

                        self.flag_record = self.flag_record_original

                    else:
                        diff = self.diff * (1 + self.diff_threshlod * 2)
                    # print("centery:{}   centerx:{}".format(self.center[0],self.center[1]))
                else:
                    new_box, t_diff, deep, color = relocate2(img_file, self.template_color, self.template_deep_feature,
                                                             self.texture,
                                                             self.template_class, self.template_box, self.color_weight1,
                                                             self.deep_feature_weight1, self.texture_weight,
                                                             center[0], center[1],
                                                             target_sz[0] * target_sz[1], distance_weight)

                    print(t_diff)

                    if t_diff != -1 and t_diff <= diff * 1.1:
                        diff = t_diff * 0.5 + self.diff * (1 + self.diff_threshlod * 2) * 0.5
                        x1 = new_box[0]
                        x2 = new_box[2]
                        y1 = new_box[1]
                        y2 = new_box[3]
                        center_y = y1 + (y2 - y1) / 2
                        center_x = x1 + (x2 - x1) / 2

                        self.center_pos = np.array([center_x, center_y])
                        bbox = [center_x - width / 2,
                                center_y - height / 2,
                                width,
                                height]

                        patch_deep_feature = deep
                        patch_color = color
                        self.flag_record = self.flag_record_original
                    else:
                        diff = self.diff * (1 + self.diff_threshlod * 2)
                        self.flag_record = self.flag_record_original

            '''阈值更新'''
            if self.flag_diff > 0:
                if idx > 5:
                    self.flag_diff = self.flag_diff - 1
                    if diff > self.diff_max:
                        self.diff_max = diff
                    self.diff_sum = self.diff_sum + diff
                if idx == 25:
                    self.diff = self.diff_max
            else:
                self.flag_diff = self.flash_num

                diff_temp = int(self.diff_sum / self.flag_diff) * 0.9 + self.diff_max * 0.1
                self.diff = int(diff_temp)

                self.diff_sum = 0
                self.diff_max = 0

            if idx % self.flash_feature == 0:
                self.template_color = self.template_color * first_weight + patch_color * recent_weight
                self.template_color = self.template_color_first * self.first_flash_weight + self.template_color * (
                        1 - self.first_flash_weight)
                self.template_deep_feature = self.template_deep_feature * first_weight + patch_deep_feature * recent_weight
                self.template_deep_feature = self.template_deep_feature * (
                        1 - self.first_flash_weight) + self.first_flash_weight * self.template_deep_feature_first

        '''#**************************************------------------------------------------------*******************'''
        width = bbox[2]
        height = bbox[3]
        cx = bbox[0] + width*0.5
        cy = bbox[1] + height*0.5
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        target_pos = self.center_pos
        target_sz = self.size
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        if self.step>1:
            z_crop = Variable(self.get_subwindow(img, target_pos, p.exemplar_size, round(s_z), avg_chans).unsqueeze(0))#[1, 3, 127, 127])
            z_crop = z_crop.float()
            z_f = self.model.template_return(z_crop.cuda())#检测模板
            z_f = torch.cat((z_f[0], z_f[1], z_f[2]), 1)  # [1, 768, 7, 7]
            if self.step==2:#模板更新方式1-Linear
                zLR=0.0102   #SiamFC[0.01, 0.05],  0.0102是siamfc初始化的方法
                z_f_ = (1-zLR) * Variable(state['z_f']).cuda() + zLR * z_f # 累积模板
                #temp = np.concatenate((init, pre, cur), axis=1)
            else:           #模板更新方式2-UpdateNet
                temp = torch.cat((Variable(state['z_0']).cuda(),Variable(state['z_f']).cuda(),z_f),1)
                init_inp = Variable(state['z_0']).cuda()
                z_f_ = self.updatenet(temp,init_inp)

            state['z_f'] = z_f_.cpu().data #累积模板
            self.model.template_back(z_f_)          #更新模板
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        self.state = state
        return state
