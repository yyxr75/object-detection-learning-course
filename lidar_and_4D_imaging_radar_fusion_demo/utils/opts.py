from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # ----------------------
        # dataset file names
        # ----------------------
        self.parser.add_argument('--training_data', default='data/datasets/oxford_dataset/dataset_filename/train.txt')
        self.parser.add_argument('--validation_data', default='data/datasets/oxford_dataset/dataset_filename/val.txt')
        self.parser.add_argument('--test_data', default='data/datasets/oxford_dataset/dataset_filename/eval.txt')
        self.parser.add_argument('--model_path', default='final_model.pth')
        self.parser.add_argument('--resume', default=False)

        # log direction
        #  training log director
        self.parser.add_argument('--output_dir', default='./outputs/logs')
        self.parser.add_argument('--experiment_name', default='0713_without_objscore')
        #  evaluation log director
        self.parser.add_argument('--map_out', default='./outputs/logs')
        self.parser.add_argument('--exp_name', default='0713_without_objscore')

        # ----------------------
        # using multi scale output for both training and validation and test
        # ----------------------
        self.parser.add_argument('--using_multi_scale', type=int, default=1,
                                 help='whether to using multi scale')
        # ----------------------
        # training params
        # ----------------------
        self.parser.add_argument('--batch_size', type=int, default=2,
                                 help='input batch size')
        self.parser.add_argument('--data_shape', type=int, nargs='+', default=[320, 320],
                                 help='the input data shape')
        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 help='start epoch')
        self.parser.add_argument('--end_epoch', type=int, default=200)
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='how many cpus')
        self.parser.add_argument('--cuda', type=int, default=1,
                                 help='1 means this machine has gpu')

        self.parser.add_argument('--rotate_angle', type=int, default=0,
                                 help='0 means ban random rotate'
                                 '1 means allow random rotate')
        # ----------------------
        # network architecture
        # ----------------------
        self.parser.add_argument('--fusion_arch', default='preCat',
                                 help='where to cat the data'
                                 'preCat: concatenate in raw spatial dimension'
                                 'afterCat: concatenate in after feature extraction')
        self.parser.add_argument('--fusion_loss_arch', default='fusion_loss_base',
                                 help='fusion_loss_base'
                                 'multi_angle_bin_loss'
                                 'rotated_iou_loss'
                                 'fusion_loss_gaussian'
                                 'anchor_angle_loss')
        # ----------------------
        # loss parameters
        # ----------------------
        self.parser.add_argument('--binNum', type=int, default=8,
                                 help='multi bin angle loss encode number, work with multi angle bin loss')
        # ----------------------
        # debug parameters
        # ----------------------
        self.parser.add_argument('--showHeatmap', type=int, default=0,
                                 help='whether to show heatmap')
        self.parser.add_argument('--chooseLoss', type=int, default=0,
                                 help='0: normal l1 loss'
                                      '1: gaussian wasserstein distance loss'
                                      '2: gaussian k-l divergence loss')

        # ----------------------
        # training parameters
        # ----------------------
        # learning rate scheduler
        self.parser.add_argument('--lr_scheduler', default='warmup_multistep')
        self.parser.add_argument('--base_lr', type=float, default=0.01,
                                 help='base learning rate')

        self.parser.add_argument('--min_lr_ratio', type=float, default=0.05)
        self.parser.add_argument('--warmup_epochs', type=int, default=2)
        self.parser.add_argument('--warmup_lr', type=float, default=0)

        # optimizer parameters
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                                 help='base learning rate')

        # multi gpu training
        self.parser.add_argument('--distributed', type=bool, default=False,
                                 help='whether use multi gpu training: True, False')
        self.parser.add_argument('--from_distributed', type=bool, default=True,
                                 help='whether read model from distributed training')
        self.parser.add_argument('--main_gpuid', type=int, default=0,
                                 help='main gpu id: 0, 1, 2 ...')
        self.parser.add_argument('--local_rank', type=int, help="local gpu id")
        self.parser.add_argument('--world_size', type=int, help="num of processes")

        # evaluator parameters
        self.parser.add_argument('--mAP_type', default='pycoco',
                                 help='choose mAP calculator: '
                                      'norm0.5'
                                      'pycoco')

        self.parser.add_argument('--heatmap_thresh', type=float, default=0.2)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt
