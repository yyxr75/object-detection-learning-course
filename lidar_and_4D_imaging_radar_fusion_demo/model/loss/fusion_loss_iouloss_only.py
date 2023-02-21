# -- coding: utf-8 --
from .fusion_loss_base import *
eps = 1e-8


class iou_loss(nn.Module):
    def __init__(self):
        super(iou_loss, self).__init__()
        # 这里我感觉应该取所有的feature map点作为生成box的anchor point
        # self.iou_calculator = calc_mAP(thresh=0.2, cfar_type='staticThresh')
        self.bbox_generator = multi_channel_object_decode(cfar_type='staticThresh')
        self.iou_calculator = calc_mAP()

    def forward(self, pred_map, gt_boxes, mask, scale, logger, opts):
        # 这里的逻辑应该是在gt anchor point的pixel位置计算boxes iou
        # pred_boxes: x,y,p,w,h,angle,p,class:0/1
        pred_boxes,_ = self.bbox_generator.gene_objects(pred_map, scale, opts)
        # pr_table: N*[conf, iou]
        pr_table, num_fn = self.iou_calculator.calc_pr_table(pred_boxes, gt_boxes, pr_table_flag=False)
        # import pdb;pdb.set_trace()
        if pr_table.shape[0] == 0:
            return 0
        iou_loss = pr_table[...,1].sum()
        # 由于这里仅仅采用了gt point位置的信息计算pred box，故这两个box数量是绝对一致的
        number = pr_table.shape[0] if pr_table.shape[0]<gt_boxes.shape[0] else gt_boxes.shape[0]
        iou_loss = iou_loss/(number+eps)
        return iou_loss


class fusion_loss_iouloss_only(fusion_loss_base):
    def __init__(self):
        super().__init__()
        self.iou_loss = iou_loss()
        pass

    def forward(self, prediction, groundTruth, opt):
        # prediction: (location, x_offset, y_offset, W, H, orientation[bin1,bin2,sin,cos], class[car,background])
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
            gt_map = torch.from_numpy(gt_map).type(torch.FloatTensor).cuda()
            gt_mask = torch.from_numpy(gt_mask).type(torch.FloatTensor).cuda()
            # -----------------------------#
            # key point heatmap loss
            # -----------------------------#
            pred_heatmap = torch.sigmoid(prediction[i,0,:,:])
            gt_heatmap = gt_map[0,0,:,:]
            loss_arr[0,1] += self.heatmap_loss(pred_heatmap, gt_heatmap)
            # -----------------------------#
            # class loss
            # [bin1,bin2]
            # -----------------------------#
            pred_cls_map = prediction[i,9:11,:,:]
            gt_cls_map = gt_map[0,8,:,:]
            loss_arr[0,5] += 0.1*self.cls_loss(pred_cls_map, gt_cls_map, gt_mask)
            # -----------------------------#
            # iou loss
            # -----------------------------#
            loss_arr[0,6] += self.iou_loss(prediction[i,...], batch_target, gt_mask, scale, opt)
        loss_arr[0, 0] += loss_arr[0, 1:].sum()
        loss_arr = loss_arr/bs
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        return loss_arr
