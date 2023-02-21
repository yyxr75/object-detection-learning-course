# -- coding: utf-8 --

# 来自yangxue的工作: https://www.zhihu.com/people/flyyoung-68
# gaussian wassserstein distance: https://arxiv.org/abs/2101.11952
# gaussian kullback-leibler divergence distance:https://arxiv.org/abs/2106.01883
# github repository: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kld_reppoints_loss.py

from .fusion_loss_base import *
from evaluator.utils_mAP import gaussian_decoder
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment

eps = 1e-8

@numba.jit(nopython=True)
def gene_gaussianMap(gaussian_map, mu, conv):
    covmat_inv = np.linalg.inv(conv).astype(np.float64)
    [width, height] = gaussian_map.shape
    for i in range(height):
        for j in range(width):
            arr = np.array([[j], [i]]).astype(np.float64)
            # import pdb;pdb.set_trace()
            v_exp = -np.dot((arr-mu).T, covmat_inv)
            v_exp = np.dot(v_exp, arr-mu)
            value = np.exp(v_exp)
            value = value[0,0]
            if value > gaussian_map[i,j]:
                gaussian_map[i,j] = value


@numba.jit(nopython=True)
def gene_casualGaussianDist(bbox, gaussian_map, scale):
    '''
    :param bbox:            4*[x,y]
    :param gaussian_map:    input & output
    :param scale:           scale
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
    centers = np.array([[center_x],[center_y]])/scale
    # centers = np.round(centers, 0, centers).astype(np.float64)
    centers = np.floor(centers)
    # import pdb;pdb.set_trace()
    u = corners - centers
    covmat = np.dot(u, u.T)/(4-1)
    gene_gaussianMap(gaussian_map, centers, covmat)


@numba.jit(nopython=True)
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
            v = -(np.power(i-x_keypoint, 2)+np.power(j-y_keypoint, 2))/(2*sigma*sigma)
            value = np.exp(v)
            if value > gaussian_map[i, j]:
                gaussian_map[i, j] = value


class gaussian_based_loss(nn.Module):
    def __init__(self):
        super(gaussian_based_loss, self).__init__()
        self.gaussian_decoder = gaussian_decoder()
    #---------------------------------
    # 对多个真值box计算生成高斯参数
    #---------------------------------
    def guassian_param_gene(self, bboxes, scale):
        '''
        :param bboxes:  所有的真值bbox
        :param scale:   当前feature map的尺度
        :return:
        '''
        num, dim = bboxes.shape
        mu_array = torch.zeros((num, 2))
        conv_array = torch.zeros((num, 2, 2))
        new_gaussian_map = np.zeros((int(320/scale), int(320/scale)))
        for i in range(num):
            bbox = bboxes[i,...]
            center_x = bbox[1]
            center_y = bbox[2]
            box_width = bbox[3]
            box_height = bbox[4]
            angle = bbox[5]
            corners = rot2D(center_x, center_y, angle, box_width, box_height)
            corners = torch.from_numpy(corners)
            corners = corners / scale
            centers = torch.tensor([center_x, center_y]) / scale
            centers = centers
            mu_array[i,...] = centers.int()
            # mu_array[i,...] = centers
            u = corners - centers.view(2,1)
            covmat = torch.matmul(u, u.T) / (4 - 1)
            conv_array[i,...] = covmat
            # ----------------------
            # 新生成的gaussian map看看
            # ----------------------
            new_mu, new_conv = mu_array[i,...].cpu().detach().numpy().reshape((2,1)), covmat.cpu().detach().numpy().T
            try:
                gene_gaussianMap(new_gaussian_map, new_mu, new_conv)
            except:
                continue

        return mu_array.cuda(), conv_array.cuda(), torch.from_numpy(new_gaussian_map).to(torch.float32).cuda()
        # return mu_array.cuda(), conv_array.cuda()
    #---------------------------------
    # 求 wasserstein distance
    #---------------------------------
    def trace(self, A):
        return A.diagonal(dim1=-2, dim2=-1).sum(-1)

    def sqrt_newton_schulz_autograd(self, A, numIters, dtype):
        '''
        multi batch function
        :param A:           batch*conv
        :param numIters:
        :param dtype:
        :return:
        '''
        A = torch.unsqueeze(A, dim=0)
        batchSize = A.data.shape[0]
        dim = A.data.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                     repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()
        Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                     repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()

        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
        return sA

    def wasserstein_distance_sigma(self, sigma1, sigma2):
        # trace(conv1+conv2-2*(conv1^1/2*conv2*conv1^1/2)^1/2)
        wasserstein_distance_item2 = torch.matmul(sigma1, sigma1) + torch.matmul(sigma2,
                                                                                 sigma2) - 2 * self.sqrt_newton_schulz_autograd(
            torch.matmul(torch.matmul(sigma1, torch.matmul(sigma2, sigma2)), sigma1), 20, torch.FloatTensor)
        wasserstein_distance_item2 = self.trace(wasserstein_distance_item2)

        return wasserstein_distance_item2

    def calc_wasserstein_distance(self, mu1, mu2, conv1 ,conv2):
        wasserstein_distance_item1 = (mu1[0] - mu2[0]) ** 2 + (mu1[1] - mu2[1]) ** 2
        # import pdb;pdb.set_trace()
        wasserstein_distance_item2 = self.wasserstein_distance_sigma(conv1, conv2)
        wasserstein_distance = torch.max(wasserstein_distance_item1 + wasserstein_distance_item2,
                                         Variable(torch.zeros(1), requires_grad=False).cuda())
        wasserstein_distance = torch.max(torch.sqrt(wasserstein_distance + eps),
                                         Variable(torch.zeros(1), requires_grad=False).cuda())
        # wasserstein_similarity = 1 / (wasserstein_distance + 2)
        # wasserstein_loss = 1 - wasserstein_similarity
        try:
            wasserstein_loss = torch.log(wasserstein_distance + 1)
        except:
            print(wasserstein_distance)
            import pdb;pdb.set_trace()

        return wasserstein_loss

    #---------------------------------
    # 求 kullback-leibler divergence
    #---------------------------------
    def calc_kl_divergence_distance(self, mu1, mu2, conv1 ,conv2):
        """Compute Kullback-Leibler Divergence.
        Args:
            g1 (dict[str, torch.Tensor]): Gaussian distribution 1.
            g2 (torch.Tensor): Gaussian distribution 2.
        Returns:
            torch.Tensor: Kullback-Leibler Divergence.
        """
        # import pdb;pdb.set_trace()
        p_mu = mu1
        p_var = conv1
        # assert p_mu.dim() == 3 and p_mu.size()[1] == 1
        # assert p_var.dim() == 4 and p_var.size()[1] == 1
        # p_mu = p_mu.squeeze(1)
        # p_var = p_var.squeeze(1)
        t_mu, t_var = mu2, conv2
        delta = (p_mu - t_mu).unsqueeze(-1)
        t_inv = torch.inverse(t_var)
        term1 = delta.transpose(-1, -2).matmul(t_inv).matmul(delta).squeeze(-1)
        term2 = torch.diagonal(
            t_inv.matmul(p_var),
            dim1=-2,
            dim2=-1).sum(dim=-1, keepdim=True) + \
                torch.log(torch.det(t_var) / (torch.det(p_var)+eps)).reshape(-1, 1)
        kl_divergence = 0.5 * (term1 + term2) - 1
        # kl_loss = 1 - 1 / (2 + torch.sqrt(kl_divergence))
        kl_loss = torch.log(kl_divergence+1)
        if torch.isinf(kl_loss) or torch.isnan(kl_loss):
            return torch.tensor(0).cuda()
            import pdb;pdb.set_trace()
        return kl_loss

    def forward(self, pred_heatmap, gt_boxes, scale, logger, opts):
        # gaussian_heatmap = pred_heatmap[0,...]
        # offset_heatmap = pred_heatmap[1:,...]
        # 对prediction gaussian heatmap进行拟合
        # pred_mu_arr, pred_conv_arr, predGaussianMap = self.guassian_mixture_fit(pred_heatmap)
        gt_mu_arr, gt_conv_arr, gtGaussianMap = self.guassian_param_gene(gt_boxes, scale)
        pred_mu_arr, pred_conv_arr, _ = self.gaussian_decoder.guassian_mixture_fit(pred_heatmap, thresh=0.5)
        # pred_mu_arr, pred_conv_arr, _ = self.gaussian_decoder.guassian_mixture_fit(gtGaussianMap, thresh=0.5)
        # 对多个真值box计算生成高斯参数
        # gt_mu_arr, gt_conv_arr, gtGaussianMap = self.guassian_param_gene(gt_boxes, scale)
        # gt_mu_arr, gt_conv_arr = self.guassian_param_gene(gt_boxes, scale)
        # import pdb;pdb.set_trace()
        # ax1=plt.subplot(1, 2, 1)
        # plt.imshow(predGaussianMap)
        # ax1.set_title('fitted prediction map')
        # ax2=plt.subplot(1, 2, 2)
        # plt.imshow(gtGaussianMap)
        # ax2.set_title('ground truth map')
        # plt.pause(0.01)
        # 目前数据全在cuda中
        # 遍历ground truth gaussian和prediction gaussian，计算wasserstein distance表
        # import pdb;pdb.set_trace()
        num_preds = pred_mu_arr.shape[0]
        num_gts = gt_boxes.shape[0]
        gwd_match_table = torch.zeros([num_preds, num_gts])
        # import pdb;pdb.set_trace()
        for i in range(num_preds):
            for j in range(num_gts):
                # 拟合后的预测高斯
                pred_mu = torch.squeeze(pred_mu_arr[i,...])
                pred_conv = torch.squeeze(pred_conv_arr[i,...])
                gt_mu = torch.squeeze(gt_mu_arr[j,...])
                gt_conv = torch.squeeze(gt_conv_arr[j,...])
                # 计算两个gaussian参数之间的wasserstein distance
                if opts.chooseLoss == 1:
                    gwd_match_table[i, j] = self.calc_wasserstein_distance(pred_mu, gt_mu, pred_conv, gt_conv)
                elif opts.chooseLoss == 2:
                    gwd_match_table[i, j] = self.calc_kl_divergence_distance(pred_mu, gt_mu, pred_conv, gt_conv)
                # gwd_match_table[i, j] = self.calc_wasserstein_distance(pred_mu, gt_mu, pred_conv, gt_conv)

        if num_preds == 0:
            # print('no key point weight greater than 0.8')
            return torch.tensor(0).cuda()
        ############################################
        # 这个函数是以行列更小的数值为标准，更小的数值是多少，输出的匹配矩阵就是多少
        # 例如输入12×13(13*12)大小的costMat，输出就是12×2的matches
        # 计算diou时需要匹配最低diou的对象, 代表两个框更靠近
        try:
            matches = linear_sum_assignment(gwd_match_table.cpu().detach().numpy())
        except:
            import pdb;pdb.set_trace()
        ############################################
        gwd_loss = gwd_match_table[matches].sum()/(num_preds+eps)
        # import pdb;pdb.set_trace()
        return gwd_loss.cuda()


class fusion_loss_gaussian(fusion_loss_base):
    def __init__(self):
        super().__init__()
        self.gwd_loss = gaussian_based_loss()
        pass

    def forward(self, prediction, groundTruth, opt):
        # prediction: (location, x_offset, y_offset, W, H, orientation[bin1,bin2,sin,cos], class[car,background])
        [_,_,width,_] =  prediction.shape
        bs = len(groundTruth)
        scale = 320/width
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        loss_arr = torch.zeros([1,10]).cuda()
        # 为什么要分开batch，因为生成gt_map要用numba加速
        # 2022-04-02: 如果要加速loss计算时间，这里应该先生成多个batch的gaussian map，然后就可以多batch处理了
        for i in range(bs):
            # -----------------------------#
            # 从ground truth中生成多通道标注图map
            # -----------------------------#
            batch_target = groundTruth[i]
            [gt_map, gt_mask] = gene_multiScaleGTmap(batch_target, opt.sigma, scale)
            gt_map = torch.from_numpy(gt_map).type(torch.FloatTensor).cuda()
            gt_mask = torch.from_numpy(gt_mask).type(torch.FloatTensor).cuda()
            # -----------------------------#
            # key point heatmap loss
            # -----------------------------#
            pred_heatmap = torch.sigmoid(prediction[i,0,:,:])
            gt_heatmap = gt_map[0,0,:,:]
            loss_arr[0,1] += self.heatmap_loss(pred_heatmap, gt_heatmap)
            # -----------------------------#
            # x,y offset loss
            # -----------------------------#
            # 这里有个疑问，应该把offset范围限制在多少,应该是取整的损失吧:scale*[0-1]
            pred_offset_map = scale*torch.sigmoid(prediction[i,1:3,:,:])
            gt_offset_map = gt_map[0,1:3,:,:]
            loss_arr[0,2] += 0.5*self.offset_loss(pred_offset_map, gt_offset_map, gt_mask)
            # -----------------------------#
            # class loss
            # [bin1,bin2]
            # -----------------------------#
            pred_cls_map = prediction[i,9:11,:,:]
            gt_cls_map = gt_map[0,8,:,:]
            # 2022-02-25: class loss并不需要很大，它的loss对整体权重更新占一个小比重就行了
            loss_arr[0,5] += self.cls_loss(pred_cls_map, gt_cls_map, gt_mask)
            # -----------------------------#
            # gaussian based loss
            # wasserstein distance loss
            # -----------------------------#
            # 预测输入gaussian heatmap
            # offset map另算
            # import pdb;pdb.set_trace()
            loss_arr[0,6] += self.gwd_loss(pred_heatmap, batch_target, scale, logger, opt)
            # pred_offset_map = scale * torch.sigmoid(prediction[i, 1:3, :, :])
            # gaussian_offset_heatmap = torch.cat((torch.unsqueeze(pred_heatmap, 0), pred_offset_map), dim=0)
            # loss_arr[0,6] += self.gwd_loss(gaussian_offset_heatmap, batch_target, scale, logger, opt)
            # --------------
            # test, using input
            # --------------
            # import pdb;pdb.set_trace()
            # loss_arr[0,6] += self.gwd_loss(gt_map[0,0,...], batch_target, scale, logger, opt)
        # import pdb;pdb.set_trace()
        loss_arr[0, 0] += loss_arr[0, 1:].sum()
        loss_arr = loss_arr/bs
        # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
        return loss_arr


if __name__=='__main__':
    # mu1 = torch.tensor([17.9577,  3.9862]).cuda()
    # conv1 = torch.tensor([[ 4.5209e-01, -1.2331e-01], [-1.2333e-01, 1.1732e+00]]).cuda()
    # mu2 = torch.tensor([18.1646,  4.3959]).cuda()
    # conv2 = torch.tensor([[ 0.7889, -0.1788], [-0.1788, 3.6353]]).cuda()

    # mu1 = torch.tensor([18.0000, 28.0000]).cuda()
    # conv1 = torch.tensor([[1.3081, -0.2204], [-0.2204,  1.3080]]).cuda()
    # mu2 = torch.tensor([18.7820, 28.6630]).cuda()
    # conv2 = torch.tensor([[1.8144, -1.1308], [-1.1308,  2.0435]]).cuda()

    # mu1 = torch.tensor([22.0161,  4.9209]).cuda()
    # conv1 = torch.tensor([[0.9975, -0.2997], [-0.2998, 1.4859]]).cuda()
    # mu2 = torch.tensor([22.8671,  5.0279]).cuda()
    # conv2 = torch.tensor([[1.0113, -0.7189], [-0.7189, 3.3993]]).cuda()
    # calculator = gaussian_wasserstein_dist_loss()
    # loss = calculator.calc_wasserstein_distance(mu1, mu2, conv1, conv2)
    # print(loss)
    # gaussian_map1 = np.zeros((40,40))
    # gaussian_map2 = np.zeros((40,40))
    # gene_gaussianMap(gaussian_map1, mu1.cpu().detach().numpy(), conv1.cpu().detach().numpy())
    # gene_gaussianMap(gaussian_map2, mu2.cpu().detach().numpy(), conv2.cpu().detach().numpy())
    # plt.subplot(121)
    # plt.imshow(gaussian_map1)
    # plt.subplot(122)
    # plt.imshow(gaussian_map2)
    # plt.show()
    # import pdb;pdb.set_trace()

    # 新的映射函数
    min_wsd = 2.1744
    wsd_arr = np.arange(0.1,10,0.1)
    print(wsd_arr)
    y1 = np.log(wsd_arr)
    y2 = np.sqrt(wsd_arr)
    y3 = 1-1/(wsd_arr-1)
    y4 = 1-1/(wsd_arr)
    y5 = 1-1/(wsd_arr+1)
    y6 = 1-1/(np.log(wsd_arr)+1)
    y7 = 1-1/(np.sqrt(wsd_arr)+1)
    # plt.plot(wsd_arr, wsd_arr)
    plt.plot(wsd_arr, y1, 'r')
    plt.plot(wsd_arr, y2, 'b')
    plt.plot(wsd_arr, y3, 'g')
    plt.plot(wsd_arr, y4, 'yellow')
    plt.plot(wsd_arr, y5)
    plt.plot(wsd_arr, y6)
    plt.plot(wsd_arr, y7)
    plt.legend(['log','sqrt','1-1/(x-1)','1-1/x','1-1/(x+1)','1-1/(log(x)+1)','1-1/(sqrt(x)+1)'])
    plt.show()
