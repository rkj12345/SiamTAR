from re_location.NetWork.net import EfficientDetBackbone
import torch
from torch.backends import cudnn
from re_location.NetWork.Box_utils import BBoxTransform, ClipBoxes
from re_location.utils.utils import preprocess, invert_affine, postprocess,display
import cv2
from siamfc_backbone.backbones import AlexNetV1
from siamfc_backbone.heads import SiamFC
import skimage.feature
import skimage.segmentation
import math
import numpy as np

def Iou(box1,box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    if xmin2>xmax1 or ymin2>ymax1 or xmin1>xmax2 or ymin1>ymax2:
        return -1

    # 求交集部分左上角的点
    xmin = max(xmin1,xmin2)
    ymin = max(ymin1,ymin2)
    # 求交集部分右下角的点
    xmax = min(xmax1,xmax2)
    ymax = min(ymax1,ymax2)

    # 计算输入的两个矩形的面积
    s1 = (xmax1-xmin1) * (ymax1 - ymin1)
    s2 = (xmax2-xmin2) * (ymax2 - ymin2)

    #计算总面积
    s = s1 + s2
    # 计算交集
    inter_area = (xmax - xmin) * (ymax - ymin)

    iou = inter_area / (s - inter_area)
    return iou




class Net(torch.nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

def relocate(img_path, template_color,template_deep_feature, template_texture, template_class, template_box, color_weight,
                 deep_feature_weight,texture_weight, center_x, center_y, s_target,distance_weight):
    compound_coef = 0
    force_input_size = 512  # 设置图片的默认尺寸
    img_path_now = img_path

    #用于筛选预测框的阈值
    #分类阈值
    threshold = 0.2
    #定位阈值
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    #数据预处理， 返回原始图片、正则化以及padding处理后的图片、图片相关信息（用双线性插值修改图片尺寸）
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path_now, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)  # 此处的torch用于扩维(将多张图片合成一个patch)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    model = EfficientDetBackbone(compound_coef=0, num_classes=len(obj_list))
    model.load_state_dict(torch.load('H:/siam_series/SiamFCpp-pysot/re_location/pth/efficientdet-d0.pth'))

    model.requires_grad_(False)
    model.eval()


    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        #用预测的回归值+anchors合成最后的预测框
        regressBoxes = BBoxTransform()
        #对合成后的预测框进行修剪（修剪到图片内部）
        clipBoxes = ClipBoxes()

        #除了上述两项任务，还将利用分类阈值和定位阈值对预测框进行筛选（NMS也在此处执行）
        out = postprocess(x,anchors, regression, classification,
                          regressBoxes, clipBoxes,threshold, iou_threshold)

    #如果对图片进行过放缩，需要将预测框放缩到对应原图的大小
    out = invert_affine(framed_metas, out)

    from torchvision import transforms
    unloader = transforms.ToPILImage()
    image1 = unloader(ori_imgs[0])
    image1.show()
    #display(out, ori_imgs, obj_list, imshow=True, imwrite=True)


    """***********************************************************************************************************"""


    model2 = Net(
        backbone=AlexNetV1(),
        head=SiamFC(0.001))
    model2.load_state_dict(torch.load("H:/siam_series/SiamFCpp-pysot/re_location/pth/siamfc_alexnet_e50.pth", map_location=lambda storage, loc: storage))
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    model2=model2.to(device)
    model2.eval()

    w = int(template_box[2])
    h = int(template_box[3])


    #目标匹配
    res = out[0]['rois']
    class_res = out[0]['class_ids']
    if len(res)==0:
        print("类型1")
        return 0,-1,-1,-1

    box_num = len(res)
    box_final = []
    diff_sum_list = []
    distance_list=[]
    texture_list=[]
    deep_list=[]
    color_list=[]
    #照片对角线距离（用于距离归一化）
    len_img=w#math.sqrt(pow((ori_imgs[0].shape[0]),2)+pow((ori_imgs[0].shape[1]),2))

    for i in range(box_num):
        box=res[i]

        class_id = class_res[i]
        if class_id==7:
            class_id=2

        h_patch=int(box[3]-box[1])
        w_patch = int(box[2]-box[0])



        s_patch=h_patch*w_patch
        s_diff=((abs(s_patch-s_target))/s_target)

        patch_box_center_x=box[0]+w_patch*0.5
        patch_box_center_y=box[1]+h_patch*0.5


        distance_patch_template=math.sqrt(pow((patch_box_center_x-center_x),2)+pow((patch_box_center_y-center_y),2))
        distance_patch_template=distance_patch_template/len_img


        if h_patch==0 or w_patch==0:
            continue
        hw_diff = abs(h_patch / w_patch - h / w)


        if class_id==template_class and hw_diff<0.3 :

            if w_patch > w:
                x1 = int(box[0] + 0.5 * w_patch - 0.5 * w)
                x2 = int(x1 + w)
            else:
                x1 = int(box[0])
                x2 = int(box[2])

            if h_patch > h:
                y1 = int(box[1] + h_patch * 0.5 - 0.5 * h)
                y2 = int(y1 + h)
            else:
                y1 = int(box[1])
                y2 = int(box[3])


            patch_img = ori_imgs[0][y1:y2, x1:x2, :]
            patch_img = cv2.resize(patch_img, (w, h), interpolation=cv2.INTER_CUBIC)
            box_final.append(box)

            # from torchvision import transforms
            # unloader=transforms.ToPILImage()
            # image1=unloader(patch_img)
            # image1.show()


            #提取纹理特征
            patch_img1 = patch_img.copy()
            patch_center_x = int(0.5 * w)
            patch_center_y = int(0.5 * h)

            train_hist = [[], [], [],[], [], []]

            img_patch_texture1 = patch_img1[
                                 int(patch_center_y - 0.5 * h):int(patch_center_y),
                                 int(patch_center_x - 0.5 * w):int(patch_center_x)]
            img_patch_texture2 = patch_img1[
                                 int(patch_center_y - 0.5 * h):int(patch_center_y),
                                 int(patch_center_x):int(patch_center_x + 0.5 * w)]
            img_patch_texture3 = patch_img1[
                                 int(patch_center_y):int(patch_center_y + 0.5 * h),
                                 int(patch_center_x - 0.5 * w):int(patch_center_x)]
            img_patch_texture4 = patch_img1[
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
                train_hist[colour_channel+3], _ = np.histogram(img_template_texture02,
                                                             density=False,
                                                             bins=256, range=(0, 256))
            texture_patch = torch.tensor(train_hist).squeeze()
            texture_diff = int(
                sum(torch.nn.functional.pairwise_distance(template_texture, texture_patch, p=2, eps=1e-06)))




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
                sum(torch.nn.functional.pairwise_distance(template_color, patch_color, p=2, eps=1e-06)))


            # 计算deep_feature差距
            patch_img2 = cv2.resize(patch_img, (96, 96), interpolation=cv2.INTER_CUBIC)
            patch_img = torch.from_numpy(patch_img2).to(
                    device).permute(2, 0, 1).unsqueeze(0).float()
            patch_deep_feature = model2.backbone(patch_img)
            patch_deep_feature = torch.squeeze(patch_deep_feature).view(256, 4)
            deep_feature_diff= int(
                    sum(torch.nn.functional.pairwise_distance(patch_deep_feature, template_deep_feature, p=2,
                                                              eps=1e-06)))

            # 记录差距
            diff_sum1 = (deep_feature_diff * deep_feature_weight+texture_diff *texture_weight+color_diff*color_weight)

            diff_sum = int(diff_sum1 * distance_patch_template * distance_weight + diff_sum1 * (1 - distance_weight))
            distance_list.append(distance_patch_template)
            diff_sum_list.append(diff_sum)
            texture_list.append(texture_patch)
            deep_list.append(patch_deep_feature)
            color_list.append(patch_color)

            print(
                "center_x:{}  center_y:{}    patch_box_center_x:{}  patch_box_center_y:{}  distance_patch_template:{}  diff_sum:{}".format(
                    center_x, center_y, patch_box_center_x, patch_box_center_y, distance_patch_template,diff_sum))

    if len(diff_sum_list)==0:
        print("类型2")
        return 0,-1,-1,-1
    index = diff_sum_list.index(min(diff_sum_list))
    box = box_final[index]
    diff=int(min(diff_sum_list)/(1-distance_weight+distance_weight*distance_list[index]))
    return box,diff,deep_list[index],color_list[index]


