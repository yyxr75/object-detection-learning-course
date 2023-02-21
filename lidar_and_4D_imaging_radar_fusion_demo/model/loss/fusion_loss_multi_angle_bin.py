# -- coding: utf-8 --
from .fusion_loss_base import *
eps = 1e-8

def tenTotwo(number, binNum):
    #定义栈
    s = torch.zeros((binNum,))
    cnt = 0
    while number > 0:
        #余数进栈
        rem = number % 2
        s[binNum-cnt-1] = rem
        number = number // 2
        cnt += 1
    return s

def tenToGray(number, binNum):
    binary = tenTotwo(number, binNum)
    gray = torch.zeros((binNum,))
    gray[0] = binary[0]
    for i in range(1, binNum):
        if binary[i-1] == binary[i]:
            gray[i] = 0
        else:
            gray[i] = 1
    return gray

def twoToTen(twoBin, binNum):
    sum=0
    for i in range(0, binNum):
        num = int(twoBin[i]) * (2 ** (binNum - i - 1))
        sum += num
    return sum

def grayToTen(grayBin, binNum):
    graycode_len = len(grayBin)
    binay = []
    binay.append(int(grayBin[0].item()))
    for i in range(1, graycode_len):
        if grayBin[i] == binay[i - 1]:
            b = 0
        else:
            b = 1
        binay.append(b)
    return twoToTen(binay, binNum)

def multi_bin_angle_encode(angle, binNum=8):
    '''
    :param angle:   等待被编码的角度值，取值范围是[-180, 180]
    :param binNum:  dense encoding, binNum=8, resolution = 360/(2^8) = 1.40625
    :return:
    '''
    angle_res = torch.FloatTensor([180.0/(2**binNum)]).cuda()
    int_binNum = int(torch.abs(angle)/angle_res)
    # encoded_bin = tenTotwo(int_binNum, binNum)
    encoded_bin = tenToGray(int_binNum, binNum)
    angle_offset = torch.abs(angle) - int_binNum*angle_res
    if angle < 0:
        angle_offset = -angle_offset
    return encoded_bin, angle_offset

def multi_bin_angle_decode(encoded_arr, angle_offset):
    '''
    :param encoded_arr:     dense encoded bin array
    :param angle_offset:    offset
    :return:
    '''
    binNum = len(encoded_arr)
    angle_res = 180.0 / (2 ** binNum)
    int_binNum = grayToTen(encoded_arr, binNum)
    # 解码的时候，从angle offset中能看出来角度的正负
    if angle_offset < 0:
        int_binNum = -int_binNum
    angle = angle_offset + int_binNum * angle_res
    return angle

def gene_multiScaleGTmap(boxes, binNum, scale):
    '''
    argument:
        boxes: number*[id, center_x, center_y, Width, Height, Angle]
        sigma: float
        scale: float
    output:
        gt_map: 5+binNum+2 x width x height (float)
        gt_mask: 1 x width x height (float)
    '''
    width = int(320/scale)
    gt_map = torch.zeros((1, 5+binNum+3, width, width))
    [num, dim] = boxes.shape
    gt_mask = torch.zeros((width, width))
    gaussian_map = torch.zeros((width, width))
    for k in range(num):
        label_center_x = int(boxes[k,2]/scale)
        label_center_y = int(boxes[k,1]/scale)
        # keyPoints = [label_center_x, label_center_y]
        # 生成一张中心点mask图
        gt_mask[label_center_x, label_center_y] = 1
        # 生成center关键点heatmap
        # gene_normGaussianDist(keyPoints, gaussian_map, width, width, sigma)
        gene_casualGaussianDist(boxes[k, :], gaussian_map, scale, width, width)
        gt_map[0, 0, ...] = gaussian_map
        # 生成center偏移offset heatmap,有正没有负, scale*[0-1]
        gt_map[0, 1, label_center_x, label_center_y] = boxes[k,1] - scale*int(boxes[k,1]/scale)
        gt_map[0, 2, label_center_x, label_center_y] = boxes[k,2] - scale*int(boxes[k,2]/scale)
        # 生成bbox长宽heatmap
        gt_map[0, 3, label_center_x, label_center_y] = boxes[k, 3]
        gt_map[0, 4, label_center_x, label_center_y] = boxes[k, 4]
        # 生成bbox方向heatmap
        angle = boxes[k, 5]
        # offset: [-angle_res, angle_res]
        binArr, offset = multi_bin_angle_encode(angle, binNum)
        # binArr = np.zeros((8,))
        # offset = 0
        gt_map[0, 5:5 + binNum, label_center_x, label_center_y] = binArr
        gt_map[0, 5 + binNum, label_center_x, label_center_y] = offset
        # 生成object score
        gt_map[0, 5 + binNum + 1, label_center_x, label_center_y] = 1
        # 生成类别
        # [1,0] = [car, background]
        gt_map[0, 5 + binNum + 2, label_center_x, label_center_y] = 0
    return gt_map, gt_mask

# 修改自yangxue的工作
# circular coding angle loss: https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40
# dense coding angle loss: https://arxiv.org/abs/2011.09670
# 他仅仅进行了角度回归的分类，没有进一步回归offset

class multi_bin_loss(nn.Module):
    # 使用8 bin稠密编码loss，0-255
    def __init__(self):
        super(multi_bin_loss, self).__init__()
        self.focalLoss = FocalLoss()

    def forward(self, pred, gt, mask, binNum):
        # -----------------------------
        # prediction
        # -----------------------------
        angle_resolution = 180/(2**binNum)
        pred_orientationBin_map = pred[:binNum,...]
        pred_orientationOffset = pred[binNum,...]
        # normalization
        # 由于采用密集编码dense coding，所有bin加起来并不等于1，而是大于1，故只能用sigmoid
        pred_orientationBin_obj = torch.sigmoid(get_map_peak(pred_orientationBin_map, mask))
        # angle offset
        # 由于输入角度的范围是[-180,180]，因此offset范围在angle_res*[-1,1]
        pred_orientationOffset_obj = angle_resolution * (2*torch.sigmoid(get_map_peak(pred_orientationOffset, mask))-1)
        #------------------------------
        # ground truth
        #------------------------------
        gt_orientationBin_map = gt[:binNum,...]
        gt_orientationOffset = gt[binNum,:,:]
        # normalization
        gt_orientationBin_obj = get_map_peak(gt_orientationBin_map, mask)
        gt_orientationOffset = get_map_peak(gt_orientationOffset, mask)
        #------------------------------
        # focal loss
        #------------------------------
        # 计算focal loss
        orientation_bin_loss = self.focalLoss(pred_orientationBin_obj, gt_orientationBin_obj)

        orientation_offset_loss = F.smooth_l1_loss(pred_orientationOffset_obj, gt_orientationOffset, reduction='sum')

        return orientation_bin_loss, orientation_offset_loss


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()

    def forward(self, pred, gt, mask):
        pred_cls_obj = torch.softmax(get_map_peak(pred, mask), dim=1)
        gt_cls_obj = get_map_peak(gt, mask).long()
        loss = F.cross_entropy(pred_cls_obj, gt_cls_obj, reduction='sum')
        return loss+eps


class fusion_loss_multi_angle_bin(fusion_loss_base):
    def __init__(self):
        super().__init__()
        self.orientation_loss = multi_bin_loss()
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
            if batch_target.shape[0] == 0:
                # print('none objects')
                break
            [gt_map, gt_mask] = gene_multiScaleGTmap(batch_target, opt.binNum, scale)
            gt_map = gt_map.cuda()
            gt_mask = gt_mask.cuda()
            # -----------------------------#
            # key point heatmap loss
            # -----------------------------#
            pred_heatmap = torch.sigmoid(prediction[i,0,:,:])
            gt_heatmap = gt_map[0,0,:,:]
            loss_arr[0,1] += self.heatmap_loss(pred_heatmap, gt_heatmap)
            # -----------------------------#
            # x,y offset loss
            # -----------------------------#
            pred_offset_map = scale*torch.sigmoid(prediction[i,1:3,:,:])
            gt_offset_map = gt_map[0,1:3,:,:]
            loss_arr[0,2] += 0.1*self.offset_loss(pred_offset_map, gt_offset_map, gt_mask)
            # -----------------------------#
            # height, width loss
            # -----------------------------#
            pred_size_map = prediction[i,3:5,:,:]
            gt_size_map = gt_map[0,3:5,:,:]
            loss_arr[0,3] += 0.1*self.size_loss(pred_size_map, gt_size_map, gt_mask)
            # -----------------------------#
            # orientation loss
            # [bin1,bin2,sin,cos]
            # -----------------------------#
            pred_orientation_map = prediction[i,5:5+opt.binNum+1,:,:]
            gt_orientation_map = gt_map[0,5:5+opt.binNum+1,:,:]
            angle_bin_loss, angle_offset_loss = self.orientation_loss(pred_orientation_map, gt_orientation_map, gt_mask, opt.binNum)
            loss_arr[0,4] += angle_bin_loss
            loss_arr[0,5] += angle_offset_loss
            # -----------------------------#
            # class loss
            # [bin1]
            # -----------------------------#
            pred_cls_map = torch.sigmoid(prediction[i, 5 + opt.binNum + 2, :, :])
            gt_cls_map = gt_map[0, 5 + opt.binNum + 2, :, :]
            cls_loss_func = nn.BCELoss(reduction='mean')
            cls_loss = cls_loss_func(pred_cls_map, gt_cls_map)
            loss_arr[0, 7] += 0.1*cls_loss
        loss_arr[0, 0] += loss_arr[0, 1:].sum()
        loss_arr = loss_arr/bs
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        return loss_arr
