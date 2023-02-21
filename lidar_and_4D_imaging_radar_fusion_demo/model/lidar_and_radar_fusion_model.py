# -- coding: utf-8 --
import torch
from torch import nn
from torch.nn import functional as F
from model.backbone import yolo
import matplotlib.pyplot as plt

def conv2d(filter_in, filter_out, kernel_size=3, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.LeakyReLU(0.1),
    )


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels = in_channels
        # if use_norm:
        #     BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        #     Linear = change_default_args(bias=False)(nn.Linear)
        # else:
        #     BatchNorm1d = Empty
        #     Linear = change_default_args(bias=True)(nn.Linear)

        self.linear= nn.Linear(self.in_channels, self.units, bias = False)
        self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1, stride=1)

        # self.t_conv = nn.ConvTranspose2d(100, 1, (1,8), stride=(1,7))
        # 直接完成了points维度归1的操作
        # points = 34+33*2=100
        # points = n+(n-1)*2 = 3n-2  => n = (points+2)/3  要更改允许的points个数，需要满足这样的条件
        # dilation = 间隔0*2
        # 等于直接用了一个100长度对卷积核对points维度进行卷积，最后直接输出为1
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 12), stride=(1, 1), dilation=(1,3))

    def forward(self, input):
        x = self.conv1(input)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class lidar_psedo_image_generate(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        # 先做第一次特征提取
        self.PFNLayer = PFNLayer(in_channels,out_channels,True,True)

    def forward(self, x):
        # 特征提取模块
        # 用卷积核完成了maxpool类似的操作，把points维度从100转换到了1
        x = self.PFNLayer(x)
        # pillar 投影成伪2d图像
        x = x.squeeze(3)
        # 我感觉应该对，view操作按照从左到右进行了reshape
        # [batch, C, pillars] = x.shape
        # width = int(np.sqrt(pillars))
        # x = x.view(batch, C, width, width)
        return x


class radarBranch(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super(radarBranch, self).__init__()
        self.downSampling = nn.Sequential(conv2d(in_channel, 2, 3, stride=1),
                                                conv2d(2, out_channel, 1, stride=1))

    def forward(self,x):
        x = self.downSampling(x)
        return x


class fusionHead(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super(fusionHead, self).__init__()
        self.head = nn.Sequential(conv2d(in_channel,in_channel),
                                  nn.Conv2d(in_channel, out_channel, 1))

    def forward(self, x):
        x = self.head(x)
        return x


class lidar_and_radar_fusion(nn.Module):
    def __init__(self, opts):
        super().__init__()
        # feature channel:1->4
        # down sampling: 4x
        self.radar_channel = 4
        self.radar_branch = radarBranch(1,4)
        # feature channel:9->64
        # down sampling: 1x
        self.lidar_channel = 16
        self.lidar_branch = lidar_psedo_image_generate(9, self.lidar_channel, True, True)
        if opts.fusion_arch == 'preCat':
            # backbone 1
            # 名字都不能改，因为存的时候是用的字典
            self.yolo_precat = yolo.YoloDecoder(self.lidar_channel + self.radar_channel)
        elif opts.fusion_arch == 'afterCat':
            # backbone 2
            pass
            self.fusion_net = yolo.YoloDecoder_2branch(self.lidar_channel, self.radar_channel)
        # head
        # out channels: (location, x_offset, y_offset, W, H, multi bin, angle offset, objscore, class[car/background])
        if opts.fusion_loss_arch == 'multi_angle_bin_loss':
            self.head3 = fusionHead(128, 5+opts.binNum+3)
            self.head4 = fusionHead(256, 5+opts.binNum+3)
            self.head5 = fusionHead(512, 5+opts.binNum+3)
        # out channels: (location, x_offset, y_offset, W, H, orientation[bin,offset], class[bin])
        elif opts.fusion_loss_arch == 'fusion_loss_base':
            self.head3 = fusionHead(128, 8)
            self.head4 = fusionHead(256, 8)
            self.head5 = fusionHead(512, 8)
        # out channels: (location, x_offset, y_offset, W, H, orientation[bin1,angle_offset], class)*anchor_num = 8*4[0,45,90,135]
        elif opts.fusion_loss_arch == 'anchor_angle_loss':
            self.head3 = fusionHead(128, 32)
            self.head4 = fusionHead(256, 32)
            self.head5 = fusionHead(512, 32)


    def forward(self, pillars, radar_image, opts):
        ##############################
        # lidar data process
        # 进行pillar伪图像处理
        ##############################
        # 输入:lidar_pillarImg= batch*channel*pillarNumber*pointNumber
        # 输出:pillar pseudo image= batch*channel*width*height:
        # 4*64*320*320
        pillars = pillars.permute(0,3,1,2)
        [bs, channels, pillar_num, points] = pillars.shape
        [_,_,width,height] = radar_image.shape
        newPillar = self.lidar_branch(pillars)

        pillarImg = newPillar.reshape((bs, self.lidar_channel, width, height))
        pillarImg = pillarImg.permute(0,1,2,3)
        if opts.showHeatmap:
            plt.figure(1)
            ax1=plt.subplot(2, 3, 1)
            ax1.set_title('raw pillar voxel image')
            raw_pillar = pillars[0, 0, :, 0].cpu().detach().numpy().reshape((width, height))
            plt.imshow(raw_pillar)

            ax2=plt.subplot(2, 3, 2)
            ax2.set_title('after conv pillar voxel image')
            plt.imshow(pillarImg[0, 0, :, :].cpu().detach().numpy())

            ax3=plt.subplot(2,3,3)
            ax3.set_title('raw radar image')
            plt.imshow(radar_image[0, 0, :, :].cpu().detach().numpy())
        # ##############################
        # # radar data process
        # ##############################
        radardata = self.radar_branch(radar_image)
        # 在进入encoder之前将feature map 拼接到一起
        if opts.fusion_arch == "preCat":
            # import pdb;pdb.set_trace()
            # concatenate lidar pillar feature map and radar feature map 4*68*80*80
            fusion_data = torch.cat([pillarImg,radardata],axis=1)
            # input into yolo backbone to extract a feature map
            # P3: (2, 128, 40, 40)
            # P4: (2, 256, 20, 20)
            # P5: (2, 512, 10, 10)
            p3,p4,p5 = self.yolo_precat(fusion_data)
            out3 = self.head3(p3)
            out4 = self.head4(p4)
            out5 = self.head5(p5)
        elif opts.fusion_arch == "afterCat":
            # input into yolo backbone to extract a feature map
            # P3: (2, 128, 40, 40)
            # P4: (2, 256, 20, 20)
            # P5: (2, 512, 10, 10)
            p3, p4, p5 = self.fusion_net(pillarImg, radardata)
            out3 = self.head3(p3)
            out4 = self.head4(p4)
            out5 = self.head5(p5)
        return out3, out4, out5
