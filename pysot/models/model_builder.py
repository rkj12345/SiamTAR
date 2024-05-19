# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from .attention.Enhanced_multi_atten import Enhanced_multi_atten

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
        self.ema1 = Enhanced_multi_atten(256)

        self.softmax = nn.Softmax(dim=1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def template_return(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        return zf
    def template_back(self, z):
        z1 = z[:, 0:256, :, :]
        z2 = z[:, 256:512, :, :]
        z3 = z[:, 512:768, :, :]
        zf=[]
        zf.append(z1)
        zf.append(z2)
        zf.append(z3)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])  #xf[0]:[1, 256, 31, 31]   self.zf[0]:[1, 256, 7, 7]
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)  #[1, 256, 25, 25]
        features = self.ema1(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda() #[32,3,127,127]
        search = data['search'].cuda() #[32,3,256,256]
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        #len(zf)=3, zf[0]:[32, 512, 15, 15] zf[1]:[32, 1024, 15, 15] zf[2]:[32, 2048, 15, 15],分别代表resnet50第三四五层特征图输出。
        zf = self.backbone(template)
        #len(xf)=3,xf[0]:[32,512,31,31]
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf) #zf[0]:[32,256,7,7]  zf[1]   zf[2]
            xf = self.neck(xf) #xf[0]:[32,256,31,31] xf[2]   xf[3]

        #输入：[32,768,7,7]     输出:[32,768,7,7]
        ########################################此处表示互相关，update头应该放在这之前
        features = self.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features) #[10, 256, 25, 25]
        ########################################此处表示送入head头
        cls, loc, cen = self.car_head(features)
        #获取出来三分支结果后，进行后处理
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
