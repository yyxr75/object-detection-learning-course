# -- coding: utf-8 --
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from loguru import logger
from evaluator.utils_mAP import peakProcess
import numba

eps = 1e-8

def drawBbox(label_leftup_x, label_leftup_y, label_width, label_height, label_angle):
    plt.figure(1)
    plt.subplot(1,3,3)
    plt.gca().add_patch(
        patches.Rectangle((label_leftup_x, label_leftup_y), label_width, label_height,
                          # angle = 180*angle/np.pi,
                          angle=360 - label_angle,
                          edgecolor='blue',
                          facecolor='none',
                          lw=2))

def rot2D(x,y,theta,w,h):
    '''
    :param x:       center x array
    :param y:       center y array
    :param theta:   degree(-180, 180)
    :param w:       bbox width
    :param h:       bbox height
    :return:        x_arr, y_arr
    '''
    theta = torch.FloatTensor([np.pi*theta/180])
    rotMat2D = torch.FloatTensor([[torch.cos(theta),torch.sin(theta)],[-torch.sin(theta),torch.cos(theta)]])
    inputArr = torch.FloatTensor([x,y]).reshape(2,-1)
    leftUp_corners = torch.FloatTensor([-w/2,-h/2]).reshape(2,-1)
    rightUP_corners = torch.FloatTensor([w/2,-h/2]).reshape(2,-1)
    leftDown_corners = torch.FloatTensor([-w/2,h/2]).reshape(2,-1)
    rightDown_corners = torch.FloatTensor([w/2,h/2]).reshape(2,-1)
    corner_lu = torch.mm(rotMat2D,leftUp_corners)+inputArr
    corner_ru = torch.mm(rotMat2D,rightUP_corners)+inputArr
    corner_ld = torch.mm(rotMat2D,leftDown_corners)+inputArr
    corner_rd = torch.mm(rotMat2D,rightDown_corners)+inputArr
    corner_lu = corner_lu.reshape(-1)
    corner_ru = corner_ru.reshape(-1)
    corner_ld = corner_ld.reshape(-1)
    corner_rd = corner_rd.reshape(-1)
    # corners = np.zeros((2,4), dtype=np.float32)
    corners = torch.FloatTensor([corner_lu[0], corner_ru[0], corner_ld[0], corner_rd[0],corner_lu[1], corner_ru[1], corner_ld[1], corner_rd[1]]).reshape(2,-1)
    # corners = np.squeeze(corners)
    return corners

def gene_casualGaussianDist(bbox, gaussian_map, scale, width, height):
    '''
    :param bbox:            4*[x,y]
    :param gaussian_map:    input & output
    :param scale:           scale
    :param width:           box size (int)
    :param height:          box size (int)
    :return:                gaussian_map
    '''
    id = bbox[0]
    center_x = bbox[1]
    center_y = bbox[2]
    box_width = bbox[3]
    box_height = bbox[4]
    angle = bbox[5]
    corners = rot2D(center_x, center_y, angle, box_width, box_height)
    # drawBbox(corners[0,0], corners[1,0], box_width, box_height, angle)
    corners = corners/scale
    centers = torch.FloatTensor([[center_x],[center_y]])/scale
    centers = torch.floor(centers)
    u = corners - centers
    covmat = torch.mm(u, u.T)/(4-1)
    covmat_inv = torch.inverse(covmat)
    [width, height] = gaussian_map.shape
    for i in range(height):
        for j in range(width):
            arr = torch.FloatTensor([[j], [i]])
            v_exp = -torch.mm((arr-centers).T, covmat_inv)
            v_exp = torch.mm(v_exp, arr-centers)
            value = torch.exp(v_exp)
            value = value[0,0]
            if value > gaussian_map[i,j]:
                gaussian_map[i,j] = value

def gene_normGaussianDist(keypoints, gaussian_map, width, height, sigma):
    '''
    :param keypoints:   [class, x_keypoint, y_keypoint, ...]
    :param width:       int
    :param height:      int
    :param sigma:       float
    :return:            2d heatmap
    '''
    [x_keypoint, y_keypoint] = keypoints
    for i in range(width):
        for j in range(height):
            v = -(torch.pow(i-x_keypoint, 2)+torch.pow(j-y_keypoint, 2))/(2*sigma*sigma)
            value = torch.exp(v)
            if value > gaussian_map[i, j]:
                gaussian_map[i, j] = value


def gene_multiScaleGTmap(boxes, scale):
    '''
    argument:
        boxes: number*[id, center_x, center_y, Width, Height, Angle]
        sigma: float
        scale: float
    output:
        gt_map: 8 x width x height (float)
        gt_mask: 1 x width x height (float)
    '''
    width = int(320/scale)
    gt_map = torch.zeros((1, 8, width, width))
    [num, dim] = boxes.shape
    gt_mask = torch.zeros((width, width))
    gaussian_map = torch.zeros((width, width))
    for k in range(num):
        # 这是对的，因为图像先行后列，对应坐标的y和x
        label_center_x = int(boxes[k,2]/scale)
        label_center_y = int(boxes[k,1]/scale)
        # 生成一张中心点mask图
        gt_mask[label_center_x, label_center_y] = 1
        # 生成center关键点heatmap
        gene_casualGaussianDist(boxes[k, :], gaussian_map, scale, width, width)
        gt_map[0, 0, ...] = gaussian_map
        # 生成center偏移offset heatmap,有正没有负, scale*[0-1]
        # 2022-03-25: 与上面反过来了，等于是先y方向的offset，然后是x方向的offset
        gt_map[0, 1, label_center_x, label_center_y] = boxes[k,1] - scale*label_center_y
        gt_map[0, 2, label_center_x, label_center_y] = boxes[k,2] - scale*label_center_x
        # 生成bbox长宽heatmap
        gt_map[0, 3, label_center_x, label_center_y] = boxes[k, 3]
        gt_map[0, 4, label_center_x, label_center_y] = boxes[k, 4]
        # 生成bbox方向heatmap和obj score
        angle = boxes[k, 5]
        # angle = torch.abs(angle)
        if angle <= 0:                                                # [angle_bin, angle_reg]
            gt_map[0, 5:7, label_center_x, label_center_y] = torch.FloatTensor([0,torch.abs(angle)])
        else:
            gt_map[0, 5:7, label_center_x, label_center_y] = torch.FloatTensor([1,torch.abs(angle)])
        # 生成类别
        # [1,0] = [car, background]
        gt_map[0, 7, label_center_x, label_center_y] = 1
    return gt_map, gt_mask


def get_map_peak(map, index_list):
    loc_x = index_list[:,0].astype(int)
    loc_y = index_list[:,1].astype(int)
    if len(map.shape) == 3:
        obj_arr = map[:,loc_x,loc_y]
    else:
        obj_arr = map[loc_x, loc_y]
    # obj_arr: N*channel
    return obj_arr


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred      保证输入gt和pred是同样维度就可以了
      gt
    '''
    # alpha = 2
    # beta = 4
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    pos_loss = torch.log(pred+eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred+eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    if np.isnan(loss.item()):
        # 这里存在预测为nan的情况。要不然就先不管，可能是异常数据，或者是异常运算
        logger.error('pred: {}'.format(pred.max()))
        logger.error('gt: {}'.format(gt.max()))
        logger.error('pos loss: {}'.format(pos_loss))
        logger.error('neg loss: {}'.format(neg_loss))
        logger.error('num pos: {}'.format(num_pos))
        logger.error('loss all: {}'.format(loss))
        loss = 1
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)+eps


class RegLossL1(nn.Module):
    def __init__(self):
        super(RegLossL1, self).__init__()

    def forward(self, out, target, matched_pair):
        '''
        argument:
            out:    channel x width x height    (float)
            target: channel x width x height    (float)
            mask:   index x width x height      (0 or 1, float)
        return:
            smooth L1 loss  (float)
        smooth_l1_loss:
            input: N*channels(N objects)
        '''
        # 在mask位置才计算，其他位置不计算
        out = get_map_peak(out, matched_pair[0])
        target = get_map_peak(target, matched_pair[1])
        # reduction='mean'意思是每个pixel计算完loss之后求平均值
        # reduction='sum'意思是每个pixel计算完loss之后求累加值
        loss = F.smooth_l1_loss(out, target, reduction='sum')
        if np.isnan(loss.item()):
            return 0
        return loss+eps


bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="sum")


class multi_bin_loss(nn.Module):
    # 这个loss写的跟描述的不一样
    # 描述的是要分成n个相互重叠的bin
    # 但是实际上center-net只分了两个bin，每个bin里4个标量
    def __init__(self):
        super(multi_bin_loss, self).__init__()

    def forward(self, pred, gt, matched_pair):
        # matched_pair: [pred_index, gt_index]
        # prediction
        pred_obj_arr = get_map_peak(pred, matched_pair[0])
        # normalization
        # bin: 0 - 1
        pred_orientationBin_obj = pred_obj_arr[0,...]
        # angle offset: 0 - 180
        pred_orientation_obj = 180 * torch.sigmoid(pred_obj_arr[1, ...])  # sin: -1-1
        # ground truth
        gt_obj_arry = get_map_peak(gt, matched_pair[1])
        gt_orientationBin_obj = gt_obj_arry[0,...]
        gt_orientation_obj = gt_obj_arry[1,...]
        # cross entropy loss
        # pred:
        #   N*C
        # target:
        #   N: index
        orientation_classification_loss = bcewithlog_loss(pred_orientationBin_obj, gt_orientationBin_obj)
        orientation_regression_loss = F.smooth_l1_loss(pred_orientation_obj, gt_orientation_obj, reduction='sum')
        return orientation_classification_loss, orientation_regression_loss


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()

    def forward(self, pred, gt, matched_pair):
        pred_cls_obj = get_map_peak(pred, matched_pair[0])
        gt_cls_obj = get_map_peak(gt, matched_pair[1])
        loss = bcewithlog_loss(pred_cls_obj, gt_cls_obj)
        return loss+eps


def anchor_point_match(pred_heatmap, gt_mask, opts):
    from scipy.optimize import linear_sum_assignment
    peakDetector = peakProcess()
    # 分别取到peak map的峰值index
    cfar_map = peakDetector.staticThresh(pred_heatmap, opts.heatmap_thresh)
    peakMap, pred_indexes = peakDetector.peakPrune(pred_heatmap, cfar_map)
    gt_indexes = torch.nonzero(gt_mask).cpu().detach().numpy()
    pred_indexes = pred_indexes.astype(int).T
    # 利用匈牙利算法进行index匹配
    num_pred = pred_indexes.shape[0]
    num_gt = gt_indexes.shape[0]
    cost_matrix = torch.zeros(num_gt, num_pred)
    for cnt_gt in range(num_gt):
        for cnt_pred in range(num_pred):
            loc_pred = pred_indexes[cnt_pred]
            loc_gt = gt_indexes[cnt_gt]
            cost_matrix[cnt_gt][cnt_pred] = np.linalg.norm(loc_pred-loc_gt)
    # 匹配的目标是最小的cost
    # 输出结果的排序是按照第一个维度的数据来的
    matches = linear_sum_assignment(cost_matrix)
    # 输出匹配的pred与gt的index pair
    selected_pred_index = pred_indexes[matches[1]]
    # 为了应对pred数量比gt数量少的情况，把没匹配上的gt index直接塞到pred index里去
    if num_gt > len(matches[0]):
        selected_pred_index = gt_indexes
        for i in range(len(matches[1])):
            idx = matches[0][i]
            selected_pred_index[idx, ...] = pred_indexes[matches[1][i]]
    # matched_pair: [pred_indexes [N*2], gt_indexes [N*2]]
    matched_pair = [selected_pred_index, gt_indexes]
    return matched_pair


class fusion_loss_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmap_loss = FocalLoss()
        self.offset_loss = RegLossL1()
        self.size_loss = RegLossL1()
        self.orientation_loss = multi_bin_loss()
        self.cls_loss = cls_loss()
        pass

    def forward(self, prediction, groundTruth, opt):
        # prediction: (location, x_offset, y_offset, W, H, orientation[bin1,offset], class)
        [_,_,width,_] =  prediction.shape
        bs = len(groundTruth)
        scale = 320/width
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        loss_arr = torch.zeros([1,10]).cuda()
        # 为什么要分开batch，因为生成gt_map要用numba加速
        for i in range(bs):
            # -----------------------------#
            # 从ground truth中生成多通道标注图map
            # -----------------------------#
            batch_target = groundTruth[i]
            [gt_map, gt_mask] = gene_multiScaleGTmap(batch_target, scale)
            gt_map = gt_map.cuda()
            gt_mask = gt_mask.cuda()
            if gt_mask.sum() == 0:
                break
            # -----------------------------#
            # key point heatmap loss
            # -----------------------------#
            pred_heatmap = torch.sigmoid(prediction[i,0,:,:])
            gt_heatmap = gt_map[0,0,:,:]
            loss_arr[0,1] += self.heatmap_loss(pred_heatmap, gt_heatmap)
            # -----------------------------#
            # 进行mask的匈牙利的匹配
            # -----------------------------#
            matched_pair = anchor_point_match(pred_heatmap, gt_mask, opt)
            # -----------------------------#
            # x,y offset loss
            # -----------------------------#
            pred_offset_map = scale*torch.sigmoid(prediction[i,1:3,:,:])
            gt_offset_map = gt_map[0,1:3,:,:]
            loss_arr[0,2] += 0.1*self.offset_loss(pred_offset_map, gt_offset_map, matched_pair)
            # -----------------------------#
            # height, width loss
            # -----------------------------#
            pred_size_map = prediction[i,3:5,:,:]
            gt_size_map = gt_map[0,3:5,:,:]
            loss_arr[0,3] += 0.1*self.size_loss(pred_size_map, gt_size_map, matched_pair)
            # -----------------------------#
            # orientation loss
            # [bin,offset]
            # -----------------------------#
            pred_orientation_map = prediction[i,5:7,:,:]
            gt_orientation_map = gt_map[0,5:7,:,:]
            angle_bin_loss, angle_offset_loss = self.orientation_loss(pred_orientation_map, gt_orientation_map, matched_pair)
            loss_arr[0,4] += 0.1*angle_bin_loss
            loss_arr[0,5] += 0.01*angle_offset_loss
            # -----------------------------#
            # class loss
            # [bin1,bin2]
            # -----------------------------#
            pred_cls_map = prediction[i,7,:,:]
            gt_cls_map = gt_map[0,7,:,:]
            try:
                loss_arr[0,6] += 0.1*self.cls_loss(pred_cls_map, gt_cls_map, matched_pair)
            except:
                logger.error(pred_cls_map[...,gt_mask==1])
                print(gt_cls_map[...,gt_mask==1])
                import pdb;pdb.set_trace()

        loss_arr[0, 0] += loss_arr[0, 1:].sum()
        loss_arr = loss_arr/bs
        # [loss all, loss heatmap, loss offset, loss size, loss orientation, loss class]
        return loss_arr
