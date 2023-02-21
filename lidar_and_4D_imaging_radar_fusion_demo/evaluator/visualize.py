# -- coding: utf-8 --
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model.loss.fusion_loss_base import gene_multiScaleGTmap
import torch
import numpy as np
from evaluator.utils_mAP import calc_mAP, multi_channel_object_decode


def drawBbox(corners, label_width, label_height, label_angle, color='blue'):
    leftup_x, leftup_y = corners[0,0], corners[1,0]
    frontCar_line = corners[:,0:2].T
    plt.gca().add_patch(
        patches.Rectangle((leftup_x, leftup_y), label_width, label_height,
                          angle=360 - label_angle,
                          edgecolor=color,
                          facecolor='none',
                          lw=2))
    plt.gca().add_patch(
        patches.Polygon(frontCar_line, closed=False, edgecolor='green', lw=3))


def rot2D(x,y,w,h,theta):
    '''
    :param x:       center x array
    :param y:       center y array
    :param theta:   degree(0-360)
    :param w:       bbox width
    :param h:       bbox height
    :return:        x_arr, y_arr
    '''
    theta = np.pi*theta/180
    rotMat2D = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    inputArr = np.array([x,y]).reshape(2,-1)
    leftUp_corners = np.array([-w/2,-h/2]).reshape(2,-1)
    rightUP_corners = np.array([w/2,-h/2]).reshape(2,-1)
    leftDown_corners = np.array([-w/2,h/2]).reshape(2,-1)
    rightDown_corners = np.array([w/2,h/2]).reshape(2,-1)
    corner_lu = np.dot(rotMat2D,leftUp_corners)+inputArr
    corner_ru = np.dot(rotMat2D,rightUP_corners)+inputArr
    corner_ld = np.dot(rotMat2D,leftDown_corners)+inputArr
    corner_rd = np.dot(rotMat2D,rightDown_corners)+inputArr
    corner_lu = corner_lu.reshape(-1)
    corner_ru = corner_ru.reshape(-1)
    corner_ld = corner_ld.reshape(-1)
    corner_rd = corner_rd.reshape(-1)
    corners = np.array([corner_lu[0], corner_ru[0], corner_ld[0], corner_rd[0],corner_lu[1], corner_ru[1], corner_ld[1], corner_rd[1]]).reshape(2,-1)
    return corners

def gtDraw(lidarpc, radar, bboxes, color='blue'):
    if radar != []:
        plt.imshow(radar)
    if lidarpc != []:
        x_arr, y_arr = lidarpc[:,0], lidarpc[:,1]
        # show lidar point cloud
        plt.scatter(x_arr, y_arr, s=0.05, c='yellow')
    # show bbox
    ids, center_x, center_y, label_width, label_height, label_angle = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4], bboxes[:,5]
    num = len(center_x)
    for i in range(num):
        id = ids[i]
        center_x_ = center_x[i]
        center_y_ = center_y[i]
        label_w_ = label_width[i]
        label_h_ = label_height[i]
        label_a_ = label_angle[i]
        corners = rot2D(center_x_, center_y_, label_w_, label_h_, label_a_)
        drawBbox(corners, label_w_, label_h_, label_a_)

def predDraw(lidarpc, radar, bboxes, color='red'):
    if radar != []:
        plt.imshow(radar)
    if lidarpc != []:
        x_arr, y_arr = lidarpc[:,0], lidarpc[:,1]
        # show lidar point cloud
        plt.scatter(x_arr, y_arr, s=0.05, c='yellow')
    # show bbox
    center_x, center_y, label_width, label_height, label_angle = bboxes[:,0], bboxes[:,1], bboxes[:,3], bboxes[:,4], bboxes[:,5]
    num = len(center_x)
    for i in range(num):
        center_x_ = center_x[i]
        center_y_ = center_y[i]
        label_w_ = label_width[i]
        label_h_ = label_height[i]
        label_a_ = label_angle[i]
        corners = rot2D(center_x_, center_y_, label_w_, label_h_, label_a_)
        drawBbox(corners, label_w_, label_h_, label_a_, color)

def drawTxt(predBox, pr_table):
    num = predBox.shape[0]
    for i in range(num):
        x = predBox[i,0]
        y = predBox[i,1]
        iou = pr_table[i,1]
        plt.text(x, y, '{:.2f}'.format(iou), bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)


def showPredResult(raw_lidarpc, raw_radar, gt_bboxes, prediction, nms_val, opts):
    '''
    :param raw_lidarpc: Nx4
    :param raw_radar:   widthxheight
    :param gt_bboxes:   mx6
    :param prediction:  kx11
    :param nms_val:     0.2 like
    :return:            null
    '''
    ################################
    # 预处理预测值和真实值
    ################################
    [width,height] = raw_radar.shape
    [pred_width,pred_height] = torch.squeeze(prediction[0]).shape[1], prediction[0].shape[2]
    scale = width/pred_width
    [gt_map, _] = gene_multiScaleGTmap(gt_bboxes, scale)
    gt_map = np.squeeze(gt_map)
    ################################
    # 解码计算bbox
    # 通过cfar算法从keymap里找到峰值后，再用峰值修剪算法剩下一个中心点位置，然后从这个关键点位置计算bbox
    ################################
    bbox_generator = multi_channel_object_decode()
    mAP_calculator = calc_mAP()
    if opts.using_multi_scale == 1:
        # pred_objects, peakMap = bbox_generator.gene_objects(prediction[0], scale, opts)
        single_batch_output = []
        for j in range(len(prediction)):
            single_batch_output.append(prediction[j][0, ...])
        pred_objects, peakMap = bbox_generator.non_max_suppression(single_batch_output,
                                                                        [width, height], opts, 0.2)
    else:
        pred_objects, peakMap = bbox_generator.non_max_suppression(prediction, [width,height], opts, nms_val)
    gt_bboxes = gt_bboxes.cpu().detach().numpy()
    pr_table,_ = mAP_calculator.calc_pr_table(pred_objects, gt_bboxes, 0.5, pr_table_flag=False,
                                                    calc_diou_flag=False)
    ################################
    # 画关键点heatmap和box结果
    ################################
    pred_heatmap = torch.squeeze(prediction[0])[0, :, :]
    pred_heatmap = torch.sigmoid(pred_heatmap)
    pred_heatmap = pred_heatmap.cpu().detach().numpy()
    gt_heatmap = gt_map[0, :, :]

    plt.figure(1)
    ax1 = plt.subplot(2, 2, 1)
    gtDraw(raw_lidarpc, raw_radar, gt_bboxes)
    ax1.set_title('final result')
    predDraw(raw_lidarpc, raw_radar, pred_objects)
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(np.squeeze(gt_heatmap))
    ax2.set_title('ground truth heatmap')
    ax3 = plt.subplot(2, 2, 3)
    plt.imshow(peakMap)
    ax3.set_title('prediction heatmap mask')
    ax4 = plt.subplot(2, 2, 4)
    plt.imshow(pred_heatmap)
    ax4.set_title('prediction heatmap')
    return pred_objects
