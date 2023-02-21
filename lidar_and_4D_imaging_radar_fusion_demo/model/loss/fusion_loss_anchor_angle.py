# -- coding: utf-8 --
from .fusion_loss_base import *

class fusion_loss_anchor_angle(fusion_loss_base):
    def __init__(self, anchor_angle_list=[0, 45, 90, 135]):
        super().__init__()
        self.anchor_angle_list = anchor_angle_list

        self.orientation_loss = multi_bin_loss()
        self.cls_loss = cls_loss()

    def forward(self, prediction, groundTruth, opt):
        # prediction: (location, x_offset, y_offset, W, H, orientation[bin1,angle_offset], class)*anchor_num = 8*3 = 24
        [_,_,width,_] =  prediction.shape
        bs = len(groundTruth)
        anchor_num = len(self.anchor_angle_list)
        scale = 320/width
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        loss_arr = torch.zeros([1,10]).cuda()
        for anchor_num in range(len(self.anchor_angle_list)):
            anchor_angle = self.anchor_angle_list[anchor_num]
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
                pred_heatmap = torch.sigmoid(prediction[i,0+anchor_num*8,:,:])
                gt_heatmap = gt_map[0,0,:,:]
                loss_arr[0,1] += self.heatmap_loss(pred_heatmap, gt_heatmap)
                # -----------------------------#
                # x,y offset loss
                # -----------------------------#
                # 这里有个疑问，应该把offset范围限制在多少,应该是取整的损失吧:scale*[0-1]
                pred_offset_map = scale*torch.sigmoid(prediction[i,1+anchor_num*8:3+anchor_num*8,:,:])
                gt_offset_map = gt_map[0,1:3,:,:]
                loss_arr[0,2] += 0.1*self.offset_loss(pred_offset_map, gt_offset_map, gt_mask)
                # -----------------------------#
                # height, width loss
                # -----------------------------#
                if opt.chooseLoss == 0:
                    pred_size_map = prediction[i,3+anchor_num*8:5+anchor_num*8,:,:]
                    gt_size_map = gt_map[0,3:5,:,:]
                    loss_arr[0,3] += 0.1*self.size_loss(pred_size_map, gt_size_map, gt_mask)
                # -----------------------------#
                # orientation loss
                # [bin,offset]
                # -----------------------------#
                pred_orientation_map = prediction[i,5+anchor_num*8:7+anchor_num*8,:,:]
                gt_orientation_map = gt_map[0,5:7,:,:]
                angle_bin_loss, angle_offset_loss = self.orientation_loss(pred_orientation_map, gt_orientation_map, gt_mask, anchor_angle)
                loss_arr[0,4] += 0.1*angle_bin_loss
                loss_arr[0,5] += 0.01*angle_offset_loss
                # -----------------------------#
                # class loss
                # [bin]
                # -----------------------------#
                pred_cls_map = prediction[i,7+anchor_num*8,:,:]
                gt_cls_map = gt_map[0,7,:,:]
                try:
                    loss_arr[0,7] += 0.1*self.cls_loss(pred_cls_map, gt_cls_map, gt_mask)
                except:
                    logger.error(pred_cls_map[...,gt_mask==1])
                    print(gt_cls_map[...,gt_mask==1])
                    import pdb;pdb.set_trace()

        loss_arr[0, 0] += loss_arr[0, 1:].sum()
        loss_arr = loss_arr/(bs*anchor_num)
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        return loss_arr

