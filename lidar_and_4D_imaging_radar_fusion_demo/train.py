#! /usr/bin/python
# -- coding: utf-8 --
from data.datasets.oxford_dataset.oxford_dataloader import OxfordDataset, oxford_dataset_collate
from torch.utils.data import DataLoader
import numpy as np
from model.lidar_and_radar_fusion_model import lidar_and_radar_fusion
import torch.optim as optim
from utils.opts import opts
from model.loss.loss_factory import get_fusionloss
from utils.fit import fit_func
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from utils.utils import *
# 在import之后直接添加以下启用代码即可
import faulthandler
# 后边正常写你的代码
faulthandler.enable()


class trainer:
    def __init__(self, opts):

        self.opts = opts
        self.log_dir = os.path.join(opts.output_dir, opts.experiment_name)
        set_log_dir(self.log_dir)
        #------------------------------------------------------#
        #   设置一些路径
        #------------------------------------------------------#
        self.train_annotation_path = opts.training_data
        self.val_annotation_path = opts.validation_data
        #------------------------------------------------------#
        #   设置显卡
        #------------------------------------------------------#
        self.ngpus_per_node = torch.cuda.device_count()
        if opts.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(opts.local_rank)
            self.global_rank = dist.get_rank()
            if opts.local_rank == opts.main_gpuid:
                logger.info(f"[{os.getpid()}] (rank={self.global_rank}, local_rank={opts.local_rank}) training ...")
                logger.info("GPU device count : ", self.ngpus_per_node)
        else:
            opts.local_rank = opts.main_gpuid
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #------------------------------------------------------#
        #   读取训练和验证数据
        #------------------------------------------------------#
        self.train_loader, self.val_loader, self.train_sampler, self.train_all_iter, self.val_all_iter = \
            self.get_data_loader(self.train_annotation_path, self.val_annotation_path, self.ngpus_per_node, opts)
        #------------------------------------------------------#
        #   创建训练网络
        #------------------------------------------------------#
        self.model = lidar_and_radar_fusion(opts)
        self.weights_init(self.model)
        #------------------------------------------------------#
        # resume training or not
        # ------------------------------------------------------#
        if opts.resume:
            model_path = os.path.join(opts.output_dir,opts.experiment_name,opts.model_path)
            self.resume_train(model_path, opts)
        else:
            self.start_epoch = opts.start_epoch
        # 多卡平行运行
        if opts.distributed:
            self.model = DDP(self.model.to(opts.local_rank), device_ids=[opts.local_rank], output_device=opts.local_rank, find_unused_parameters=True, broadcast_buffers=False)
        else:
            self.model = self.model.to(self.device)
            cudnn.benchmark = True
        self.optimizer = optim.Adam(self.model.parameters(), opts.base_lr, weight_decay=opts.weight_decay)
        self.lr_scheduler = self.get_lr_scheduler(opts, self.train_all_iter)
        #------------------------------------------------------#
        #   创建 loss 和 log 函数
        #------------------------------------------------------#
        self.fusionLoss = get_fusionloss(opts.fusion_loss_arch)
        # ----------------------------#
        #   fit function
        # ----------------------------#
        self.fit_func = fit_func(
            self.model,
            self.fusionLoss,
            self.optimizer,
            self.train_all_iter,
            self.val_all_iter,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
            opts,
        )

    def weights_init(self, net, init_type='normal', init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
        logger.info('initialize network with %s type' % init_type)
        net.apply(init_func)

    def resume_train(self, model_path, opts):
        # load model
        if opts.local_rank == opts.main_gpuid:
            logger.info('Load weights {}.'.format(model_path))
        model_dict = self.model.state_dict()
        stat = torch.load(model_path)
        pretrained_dict = stat['model']
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if opts.distributed:
                new_key = k[7:]
            else:
                new_key = k
            if np.shape(model_dict[new_key]) == np.shape(v):
                temp_dict[new_key] = v
                load_key.append(new_key)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        self.model.load_state_dict(model_dict)
        # load epoch
        self.start_epoch = stat['epoch']
        if opts.local_rank == opts.main_gpuid:
            logger.info("\nSuccessful Load Key:", str(load_key)[:500], ".......\nSuccessful Load Key Num:", len(load_key))
            logger.info("\nFail To Load Key:", str(no_load_key)[:500], ".......\nFail To Load Key num:", len(no_load_key))
            logger.info("\n current epoch:{}".format(self.start_epoch))
            # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        # load optimizer
        optimizer = stat['optimizer']
        self.optimizer.load_state_dict(optimizer)
        # 因为optimizer加载参数时,tensor默认在CPU上
        # 故需将所有的tensor都放到cuda,
        # 否则: 在optimizer.step()处报错：
        # RuntimeError: expected device cpu but got device cuda:0
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def get_data_loader(self,
            train_annotation_path,
            val_annotation_path,
            ngpus_per_node,
            opts
    ):
        with open(train_annotation_path) as f:
            train_lines = f.readlines()
        with open(val_annotation_path) as f:
            val_lines = f.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)
        max_epoch_step = num_train // opts.batch_size
        max_epoch_step_val = num_val // opts.batch_size

        train_dataset = OxfordDataset(train_lines, opts, opts.data_shape, train=True)
        val_dataset = OxfordDataset(val_lines, opts, opts.data_shape, train=False)

        if opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            batch_size_train = opts.batch_size // ngpus_per_node
            batch_size_val = opts.batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            batch_size_train = opts.batch_size
            batch_size_val = opts.batch_size
            shuffle = True

        # num_workers 每次同时初始化这么多个线程来读取数据，在这些线程中的随机旋转角度是一样的
        train_loader = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=batch_size_train,
            num_workers=opts.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=oxford_dataset_collate,
            sampler=train_sampler
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=shuffle,
            batch_size=batch_size_val,
            num_workers=opts.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=oxford_dataset_collate,
            sampler=val_sampler
        )

        return train_loader, val_loader, train_sampler, max_epoch_step, max_epoch_step_val

    def get_lr_scheduler(self, opts, iters_per_epoch):
        from utils.lr_scheduler import WarmupMultiStepLR
        warmup_epochs = opts.warmup_epochs
        lr_scheduler = WarmupMultiStepLR(
            self.optimizer,
            milestones=[10, 20],
            warmup_factor=0.001,
            warmup_epoch=warmup_epochs,
            iters_per_epoch=iters_per_epoch
        )
        return lr_scheduler

    def train(self):
        #------------------------------------------------------#
        #   开始拟合训练
        #------------------------------------------------------#
        for epoch in range(self.start_epoch, self.opts.end_epoch):
            if self.opts.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            self.fit_func.fit(epoch)


if __name__ == "__main__":
    opt = opts().parse()
    trainer = trainer(opt)
    trainer.train()

