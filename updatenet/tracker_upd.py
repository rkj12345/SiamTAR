import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils_upd import  get_subwindow
from config_upd import Config as TrackerConfig
from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.models.model_builder import ModelBuilder
import cv2
from pysot.utils.misc import bbox_clip
from torchvision import  transforms

def _convert_cls( cls):
    cls = F.softmax(cls[:, :, :, :], dim=1).data[:, 1, :, :].cpu().numpy()
    return cls
def sz( w, h):
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))
def change(r):
    return np.maximum(r, 1. / r)

def cal_penalty(target_sz, lrtbs, penalty_lk, scale_z):
    bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
    bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
    s_c = change(sz(bboxes_w, bboxes_h) / sz(target_sz[0]*scale_z, target_sz[1]*scale_z))
    r_c = change((target_sz[0] / target_sz[1]) / (bboxes_w / bboxes_h))
    penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
    return penalty
def accurate_location( max_r_up, max_c_up):
    dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
    max_r_up += dist
    max_c_up += dist
    p_cool_s = np.array([max_r_up, max_c_up])
    disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
    return disp
def coarse_location( hp_score_up, p_score_up, scale_score, lrtbs):
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
def getCenter(scale_z,hp_score_up, p_score_up, scale_score,lrtbs,target_pos):
    # corse location
    score_up = coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
    # accurate location
    max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
    disp = accurate_location(max_r_up,max_c_up)
    disp_ori = disp / scale_z
    new_cx = disp_ori[1] + target_pos[0]
    new_cy = disp_ori[0] + target_pos[1]
    return max_r_up, max_c_up, new_cx, new_cy

def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    #cls, loc, cen = net.track(x_crop)
    #outputs['cls']:[1, 2, 25, 25]  outputs['loc']:[1, 4, 25, 25]  outputs['cen']:[1, 1, 25, 25]
    outputs = net.track(x_crop)
    cls = _convert_cls(outputs['cls']).squeeze()#(25, 25)
    cen = outputs['cen'].data.cpu().numpy()#(1, 1, 25, 25)
    cen = (cen - cen.min()) / cen.ptp()
    cen = cen.squeeze()  # (25, 25)
    lrtbs = outputs['loc'].data.cpu().numpy().squeeze()  # (4, 25, 25)
    upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1  # 193
    ########################################################################################################################
    #此处每一个数据集参数不一样，需要修改。
    #got-10k
    hp = {'lr': 0.7, 'penalty_k': 0.06, 'window_lr': 0.1}
    penalty = cal_penalty(target_sz,lrtbs, hp['penalty_k'], scale_z)  # [25,25]
    p_score = penalty * cls * cen  # (25, 25)

    if cfg.TRACK.hanming:  # 汉宁窗惩罚，
        hp_score = p_score * (1 - hp['window_lr']) + window * hp['window_lr']## (25, 25)
    else:
        hp_score = p_score

    hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # (193, 193)
    p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # (193, 193)
    cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # (193, 193)
    lrtbs = np.transpose(lrtbs, (1, 2, 0))  # (25, 25, 4)
    lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # (193, 193, 4)

    scale_score = upsize / cfg.TRACK.SCORE_SIZE
    # get center
    max_r_up, max_c_up, new_cx, new_cy = getCenter(scale_z,hp_score_up, p_score_up, scale_score, lrtbs,target_pos)
    # get w h
    ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / scale_z
    ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / scale_z

    s_c = change(sz(ave_w, ave_h) / sz(target_sz[0] * scale_z, target_sz[1] * scale_z))
    r_c = change((target_sz[0] / target_sz[1]) / (ave_w / ave_h))
    penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])  # 0.8779869035415468
    lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
    new_width = lr * ave_w + (1 - lr) * target_sz[0] #差距太大
    new_height = lr * ave_h + (1 - lr) * target_sz[1] #差距太大

    # clip boundary
    # cx = bbox_clip(new_cx, 0, img.shape[1])
    # cy = bbox_clip(new_cy, 0, img.shape[0])
    # width = bbox_clip(new_width, 0, img.shape[1])
    # height = bbox_clip(new_height, 0, img.shape[0])

    # udpate state
    target_pos = np.array([new_cx, new_cy])
    target_sz = np.array([new_width, new_height])
    return target_pos, target_sz


def SiamCAR_init_upd(im, target_pos, target_sz, net):


    state = dict()
    p = TrackerConfig()
    #p.update(net.cfg)
    # p.update(net.cfg)
    state['im_h'] = im.shape[0]  # 原图的长宽
    state['im_w'] = im.shape[1]
    if p.adaptive:  # GT框的面积与原图面积之比
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 255


    # p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1  # 271-127/8+1=19

    # p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))  # (19*19*5,4)

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)

    # s_z = round(np.sqrt(wc_z * hc_z))#python2
    s_z = round(np.sqrt(wc_z * hc_z))  # python3和python2Round

    # initialize the exemplar
    # [3, 127, 127],  im:原图，target_pos:gt框中心坐标
    z_crop = get_subwindow(im, target_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    round(s_z), avg_chans )

    # [1, 3, 127, 127]
    z = Variable(z_crop.unsqueeze(0))  # [1, 3, 127, 127]
    z = z.float()
    net.template(z.cuda())
    z_f = net.template_return(z.cuda())# len(z_f)=3,分别为resnet50提取出来的三四五层特征。
    #print(111111)
    z_f = torch.cat((z_f[0], z_f[1], z_f[2]), 1) #[1, 768, 7, 7]
    #print(111111)
    #net.template(z_f)

    # if p.windowing == 'cosine':
    #     # [19,19]
    #     window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    # elif p.windowing == 'uniform':
    #     window = np.ones((p.score_size, p.score_size))
    # # (1805,)
    # window = np.tile(window.flatten(), p.anchor_num)
    hanning = np.hanning(cfg.TRACK.SCORE_SIZE)#(25,)
    window = np.outer(hanning, hanning)#(25, 25)
    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['z_f_cur'] = z_f.cpu().data  # 当前检测特征图 #[1, 768, 7, 7]
    state['z_f'] = z_f.cpu().data  # 累积的特征图
    state['z_0'] = z_f.cpu().data  # 初始特征图
    state['gt_f_cur'] = z_f.cpu().data  # gt框对应的特征图
    return state
def SiamCAR_track_upd(state, im, updatenet):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        gt_pos = state['gt_pos']
        gt_sz = state['gt_sz']
        window = state['window']

        wc_z = gt_sz[0] + p.context_amount * sum(gt_sz)
        hc_z = gt_sz[1] + p.context_amount * sum(gt_sz)
        s_z = np.sqrt(wc_z * hc_z)

        gt_crop = get_subwindow(im, gt_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    round(s_z), avg_chans)
        gt_crop = Variable(gt_crop.unsqueeze(0))  # [1, 3, 127, 127]
        g_f = gt_crop.float()
        g_f = net.backbone(g_f.cuda())  # len(z_f)=3,分别为resnet50提取出来的三四五层特征。

        g_f = net.neck(g_f)
        g_f = torch.cat((g_f[0], g_f[1], g_f[2]), 1)  #[1, 768, 7, 7]


        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)

        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        d_search = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE ) // 2

        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        #在前一个目标位置提取搜索区域x的缩放裁剪图 ([3, 255, 255])
        x_crop = get_subwindow(im, target_pos,
                                cfg.TRACK.INSTANCE_SIZE,
                                round(s_x), avg_chans)
        x_crop = Variable(x_crop.unsqueeze(0))
        x_crop = x_crop.float()
        #target——pos预测出来的相差不大，target_sz,也就是宽高差别大，是否需要调参？
        target_pos, target_sz = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window,
                                                    scale_z, p)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        z_crop = get_subwindow(im, target_pos, p.exemplar_size, round(s_z), avg_chans)

        toPIL = transforms.ToPILImage()
        pic = toPIL(z_crop)
        z_crop = Variable(z_crop.unsqueeze(0))#[1, 3, 127, 127]
        z_crop = z_crop.float()
        #net.template(z_crop)
        z_f = net.template_return(z_crop)
        # print(111111)
        z_f = torch.cat((z_f[0], z_f[1], z_f[2]), 1)  # [1, 768, 7, 7]
        # 模板更新方式1-Linear
        if updatenet == '':
            zLR = 0.0102  # SiamFC默认的更新频率
            z_f_ = (1 - zLR) * Variable(state['z_f']).cuda() + zLR * z_f  # 累积模板,[1, 768, 7, 7]
        # temp = np.concatenate((init, pre, cur), axis=1)

        # 模板更新方式2-UpdateNet
        else: #state['z_0']:[1, 768, 7, 7]    state['z_f']:[1, 768, 7, 7]     z_f: [1, 768, 7, 7]
            temp = torch.cat((Variable(state['z_0']).cuda(), Variable(state['z_f']).cuda(), z_f), 1) #[1, 2304, 7, 7]
            init_inp = Variable(state['z_0']).cuda() #[1, 768, 7, 7]
            z_f_ = updatenet(temp, init_inp)  # 累积特征图
        net.template_back(z_f_)
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['z_f'] = z_f_.cpu().data  # 累积模板
        state['z_f_cur'] = z_f.cpu().data  # 当前检测模板
        state['gt_f_cur'] = g_f.cpu().data  # 当前帧gt框对应的特征模板
        state['net'] = net
        return state