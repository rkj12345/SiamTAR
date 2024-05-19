from re_location.NetWork.net import EfficientDetBackbone
import torch
from torch.backends import cudnn
from re_location.NetWork.Box_utils import BBoxTransform, ClipBoxes
from re_location.utils.utils import preprocess, invert_affine, postprocess,display


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




def detect(img_path,box_t):
    compound_coef = 0
    force_input_size = 512  # 设置图片的默认尺寸

    #用于筛选预测框的阈值
    #分类阈值
    threshold = 0.2
    #定位阈值
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    input_sizes = [256, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


    #数据预处理， 返回原始图片、正则化以及padding处理后的图片、图片相关信息（用双线性插值修改图片尺寸）
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path,max_size=input_size)


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
    model.load_state_dict(torch.load('D:\\codeProjects\\SiamCAR_3080_checkpot_e19-master-updateneting\\re_location\\pth\\efficientdet-d0.pth'))

    model.requires_grad_(False)
    model.eval()


    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # with torch_no_grad( )：数据不需要梯度计算，即不会进行反相传播;
    # net.eval( )：不加的话即使没有训练输入数据也会改变权值，因为这是禁止forward过程对
    # 参数造成的影响;例如禁止dropout或者因为测试集和训练集的样本分布不一样，会有batch normalization 所带来的影响。

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

    cudnn.fastest = False
    cudnn.benchmark = False
    '''**************************************************************************************************************'''


    box_template=[box_t[0],box_t[1],box_t[0]+box_t[2],box_t[1]+box_t[3]]

    s_template = box_t[3]*box_t[2]

    res = out[0]['rois']
    class_res=out[0]['class_ids']
    box_num = len(res)
    box_final = []
    threshold_s_bigger =4
    threshold_s_smaller = 0.75
    class_template = []
    iou_list=[]

    if box_num<1:
        return -1, 0, 0, 0 , 0

    for i in range(box_num):
        patch_box = res[i]
        class_id=class_res[i]

        h_patch = int(patch_box[3] - patch_box[1])
        w_patch = int(patch_box[2] - patch_box[0])
        s_patch = h_patch * w_patch
        s_diff = float(abs(s_patch - s_template) / s_template)

        if h_patch == 0 or w_patch == 0:
            continue

        if (s_diff<threshold_s_bigger and s_patch>=s_template) or (s_diff<threshold_s_smaller and s_patch<s_template):
            iou=Iou(patch_box,box_template)

            if iou>0.1:
                iou_list.append(iou)
                box_final.append(patch_box)
                class_template.append(class_id)
                box_final.append(patch_box)

                patch_img=ori_imgs[0][int(patch_box[1]):int(patch_box[3]),int(patch_box[0]):int(patch_box[2])]
                # from torchvision import transforms
                # unloader=transforms.ToPILImage()
                # image1=unloader(patch_img)
                # image1.show()

    if len(iou_list)==0:
        return -1, 0, 0, 0 , 0
    else:
        index = iou_list.index(max(iou_list))
        class_result= class_template[index]
        box=box_final[index]
        template_h=box[3]-box[1]
        template_w=box[2]-box[0]
        if class_result == 7:
            class_result = 2
        print(obj_list[class_result])
        return class_result, template_h, template_w,box[1],box[0]

