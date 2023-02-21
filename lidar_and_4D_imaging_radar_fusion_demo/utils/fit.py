# -- coding: utf-8 --
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from evaluator.utils_mAP import multi_channel_object_decode
from pycocotools.coco import COCO
from collections import defaultdict
import copy
from evaluator.MVDNet_mAPtools import RobotCarCOCOeval
from loguru import logger
from utils.allreduce_norm import all_reduce_norm
from utils.dist import gather, synchronize
import os
import itertools

class fit_func():
    def __init__(self,
                 model,
                 fusionLoss,
                 optimizer,
                 train_all_iter,
                 val_all_iter,
                 train_loader,
                 val_loader,
                 lr_scheduler,
                 opts,
    ):
        self.model = model
        self.fusionLoss = fusionLoss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_all_iter = train_all_iter
        self.val_all_iter = val_all_iter
        self.iter = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opts = opts
        self.log_path = os.path.join(opts.output_dir, opts.experiment_name)
        self.start_epoch = opts.start_epoch
        self.end_epoch = opts.end_epoch

        # 创建bbox解码器
        self.bbox_generator = multi_channel_object_decode()

        logger.info("args: {}".format(opts))

    def save_checkpoint(self, state, save_dir, model_name="final_model"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        torch.save(state, filename)

    def save_model(self, model_name, epoch, ap_50):
        state = {
            "epoch": epoch + 1,
            "model": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ap_50': ap_50,
        }
        self.save_checkpoint(state, self.log_path, model_name)

    def append_loss(self, train_loss, val_mAP):
        with open(os.path.join(self.opts.output_dir, self.opts.experiment_name, "epoch_loss" + ".txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(os.path.join(self.opts.output_dir, self.opts.experiment_name, "epoch_val_mAP" + ".txt"), 'a') as f:
            f.write(str(val_mAP))
            f.write("\n")

    def loss_log(self, loss_arr):
        loss_dict = {
            'loss_all': loss_arr[0, 0].item(),
            'heatmap_loss': loss_arr[0, 1].item(),
            'xy_offset_loss': loss_arr[0, 2].item(),
            'wh_loss': loss_arr[0, 3].item(),
            'angle_bin': loss_arr[0, 4].item(),
            'angle_offset': loss_arr[0, 5].item(),
            'cls_loss': loss_arr[0, 6].item()
        }
        loss_str = ", ".join(
            ["{}: {:.3f}".format(k, v) for k, v in loss_dict.items()]
        )
        if self.opts.local_rank == self.opts.main_gpuid:
            logger.info(loss_str)

    def creat_coco(self):
        # ----------------------#
        # 创建coco mAP对象
        # ----------------------#
        coco_gt = COCO()
        coco_gt.dataset = dict()
        coco_gt.anns = dict()
        coco_gt.cats = dict()
        coco_gt.imgs = dict()
        coco_gt.imgToAnns = defaultdict(list)
        coco_gt.catToImgs = defaultdict(list)
        coco_gt.dataset["images"] = []

        coco_gt.dataset["categories"] = []
        category = dict()
        category["supercategory"] = "vehicle"
        category["id"] = 1
        category["name"] = "car"
        coco_gt.dataset["categories"].append(category)
        coco_gt.dataset["annotations"] = []

        return coco_gt

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def progress_in_iter(self):
        return self.epoch * self.train_all_iter + self.iter

    def training_step(self, epoch):
        ################
        # lidarpc:      batch_size*点云
        # radardata:    batch_size*毫米波雷达heatmap
        # boxes:        batch_size*label框
        # labels        batch_size*label类别
        ################
        # batch*pillars*N*D
        self.model.train()
        all_loss_train = torch.zeros((1,10)).cuda()
        for self.iter, batch in enumerate(self.train_loader):
            if self.iter >= self.train_all_iter:
                break
            lidar_pillar, radardata, boxes, labels = batch[0], batch[1], batch[2], batch[3]
            radardata = np.expand_dims(radardata, axis=1)
            lidar_pillar = torch.from_numpy(lidar_pillar).type(torch.FloatTensor).cuda()
            radardata = torch.from_numpy(radardata).type(torch.FloatTensor).cuda()
            boxes = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in boxes]
            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播
            # 3个尺度下的输出
            output = self.model(lidar_pillar, radardata, self.opts)
            # -------------------
            # 计算损失
            # -------------------
            # loss all, loss heatmap, loss offset, loss size, loss orientation, loss class, loss iou, ...
            loss_train_arr = torch.zeros([1, 10]).cuda()
            # 是否使用多尺度
            if self.opts.using_multi_scale == 1:
                num_output = len(output)
            else:
                num_output = 1
            # 多卡训练必须将所有计算出来的全部计算loss，不然就报错
            for i in range(num_output):
                loss_train_arr += self.fusionLoss(output[i], boxes, self.opts)
            loss_train_arr = loss_train_arr / num_output
            # ----------------------#
            #   反向传播
            # ----------------------#
            # 多卡训练时可能会在loss=0处进程阻塞，所以跳过loss=0的backward
            if loss_train_arr[0, 0].item() != 0:
                loss_train_arr[0, 0].backward()
            self.loss_log(loss_train_arr)
            if self.opts.local_rank == self.opts.main_gpuid:
                progress_str = "epoch: {}/{}, iter: {}/{} optim_lr: {}, sche_lr: {}".format(
                    epoch + 1, self.end_epoch, self.iter + 1, self.train_all_iter, self.optimizer.param_groups[0]['lr'], self.get_lr()
                )
                # import pdb;pdb.set_trace()
                # logger.info(self.optimizer.param_groups[0]['params'][0])
                logger.info("{}".format(progress_str))
            self.optimizer.step()
            self.lr_scheduler.step()
            # # update lr
            # lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            # # update optimizer
            # for param_group in self.optimizer.param_groups:
            #     param_group["lr"] = lr
            # # error process
            if np.isnan(loss_train_arr[0,0].item()):
                continue
            # collect all loss
            all_loss_train += loss_train_arr
        # get average loss
        all_loss_train = all_loss_train/self.train_all_iter
        self.loss_log(all_loss_train)

        return all_loss_train[0,0]

    def validation_step(self, epoch):
        ################
        # lidarpc:      batch_size*点云
        # radardata:    batch_size*毫米波雷达heatmap
        # gt_boxes:        batch_size*label框
        # labels        batch_size*label类别
        ################
        # batch*pillars*N*D
        self.model.eval()
        # 创建coco mAP对象
        coco_gt = self.creat_coco()
        coco_results = []
        gt_objNum = 0
        # 创建假的timestamp和fram number
        timestamp = 0
        frame_num = 0
        # ----------------------#
        # 开始计算
        # ----------------------#
        for iter_val, batch in enumerate(self.val_loader):
            if iter_val >= self.val_all_iter:
                break
            lidar_pillar, radardata, boxes, labels = batch[0], batch[1], batch[2], batch[3]
            if self.opts.showHeatmap == True:
                plt.figure(1)
                plt.imshow(np.squeeze(radardata[0]))
            radardata = np.expand_dims(radardata, axis=1)
            lidar_pillar = torch.from_numpy(lidar_pillar).type(torch.FloatTensor).cuda()
            radardata = torch.from_numpy(radardata).type(torch.FloatTensor).cuda()
            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播
            # 3个尺度下的输出
            output = self.model(lidar_pillar, radardata, self.opts)
            # 计算损失
            [width, height] = radardata.shape[2], radardata.shape[3]
            output_singleScale = output[0]
            batch_size, channels, pred_width, pred_height = output_singleScale.shape
            for batch in range(batch_size):
                # ---------------------
                # 解码计算bbox
                # ---------------------
                gt_boxes_singleBatch = boxes[batch]
                scale = width / pred_width
                # -------------------
                # 是否使用多尺度
                # -------------------
                if self.opts.using_multi_scale == 1:
                    single_batch_output = []
                    for j in range(len(output)):
                        single_batch_output.append(output[j][batch, ...])
                    pred_objects, peakMap = self.bbox_generator.non_max_suppression(single_batch_output,
                                                                               [width, height], self.opts, 0.2)
                else:
                    output_singleBatch = torch.squeeze(output_singleScale[batch, ...])
                    pred_objects, peakMap = self.bbox_generator.gene_objects(output_singleBatch, scale, self.opts)
                # -------------------
                # 使用pycopcotools的mAP计算器
                # -------------------
                if self.opts.mAP_type == "pycoco":
                    timestamp += 1
                    frame_num += 1

                    image = dict()
                    image["timestamp"] = timestamp
                    image["id"] = frame_num
                    coco_gt.dataset["images"].append(image)

                    for i in range(gt_boxes_singleBatch.shape[0]):
                        gt_objNum += 1
                        coco_ann = dict()
                        coco_ann["area"] = gt_boxes_singleBatch[i, 3] * gt_boxes_singleBatch[i, 4]
                        coco_ann["iscrowd"] = 0
                        coco_ann["image_id"] = frame_num
                        coco_ann["bbox"] = copy.deepcopy(
                            [gt_boxes_singleBatch[i, 1], gt_boxes_singleBatch[i, 2], gt_boxes_singleBatch[i, 3],
                             gt_boxes_singleBatch[i, 4], gt_boxes_singleBatch[i, 5]])
                        coco_ann["category_id"] = 1
                        coco_ann["id"] = gt_objNum
                        coco_gt.dataset["annotations"].append(coco_ann)

                    for i in range(pred_objects.shape[0]):
                        # mvdnet angle = (true angle + 180) (+- 360)
                        new_angle = pred_objects[i, 5] + 180
                        if new_angle > 180:
                            new_angle -= 360
                        elif new_angle < -180:
                            new_angle += 360
                        result = dict()
                        result["image_id"] = frame_num
                        result["category_id"] = 1
                        result["bbox"] = copy.deepcopy(
                            [pred_objects[i, 0], pred_objects[i, 1], pred_objects[i, 3], pred_objects[i, 4], new_angle])
                        result["score"] = pred_objects[i, 2]
                        coco_results.append(result)
                # 显示
                if self.opts.local_rank == self.opts.main_gpuid:
                    progress_str = "epoch: {}/{}, iter: {}/{}".format(
                        epoch + 1, self.end_epoch, iter_val + 1, self.val_all_iter
                    )
                    frame_counter_str = ' frame number: {}'.format(frame_num)
                    pred_obj_num_str = ' prediction object number: {}'.format(len(coco_results))
                    gt_obj_num_str = ' ground truth object number: {}'.format(gt_objNum)
                    logger.info(progress_str+frame_counter_str+pred_obj_num_str+gt_obj_num_str)

        # all_result_bbox = gather(coco_results)
        # all_result_bbox = all_result_bbox[0]

        if self.opts.distributed:
            synchronize()
            coco_results = gather(coco_results, dst=0)
            coco_results = list(itertools.chain(*coco_results))
            logger.info("all gathered evaluation object number:{}".format(len(coco_results)))
        if self.opts.local_rank != self.opts.main_gpuid or len(coco_results) == 0:
            logger.info('current process id is {}, skip coco evaluation! '.format(self.opts.local_rank))
            return 0
        coco_gt.createIndex()
        # --------------------------
        # 调用coco api 进行mAP估计
        # --------------------------
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = RobotCarCOCOeval(coco_gt, coco_dt, iouType="bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_50 = coco_eval.stats[1]
        synchronize()

        return ap_50

    def fit(self, epoch):
        self.epoch = epoch
        # ----------------------#
        #   training step
        # ----------------------#
        if self.opts.local_rank == self.opts.main_gpuid:
            logger.info('start training!')
        all_loss = self.training_step(epoch)
        # ----------------------#
        #   validation
        # ----------------------#
        if self.opts.local_rank == self.opts.main_gpuid:
            logger.info('Finish Train')
            logger.info('Start validation')
        with torch.no_grad():
            ap_50 = self.validation_step(epoch)
        if self.opts.local_rank == self.opts.main_gpuid:
            logger.info('Finish Validation')
        # ---------------------------
        # 保存模型和loss
        # ---------------------------
        if self.opts.local_rank == self.opts.main_gpuid:
            self.append_loss(all_loss.item(), ap_50)
            model_name = "epoch{:03d}-train_{:.3f}-val_{:.3f}".format(epoch, all_loss, ap_50)
            self.save_model(model_name, epoch, ap_50)
