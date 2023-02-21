# -- coding: utf-8 --
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .fusion_loss_iouloss_only import fusion_loss_iouloss_only
from .fusion_loss_multi_angle_bin import fusion_loss_multi_angle_bin
from .fusion_loss_gaussian import fusion_loss_gaussian
from .fusion_loss_base import fusion_loss_base
from .fusion_loss_anchor_angle import fusion_loss_anchor_angle


# dataset_factory = {
#     'coco': COCO,
#     'pascal': PascalVOC,
#     'kitti': KITTI,
#     'coco_hp': COCOHP
# }

loss_factory = {
    'fusion_loss_base': fusion_loss_base,
    'rotated_iou_loss': fusion_loss_iouloss_only,
    'multi_angle_bin_loss': fusion_loss_multi_angle_bin,
    'fusion_loss_gaussian': fusion_loss_gaussian,
    'anchor_angle_loss': fusion_loss_anchor_angle,
}


def get_fusionloss(loss_type):
    fusion_loss = loss_factory[loss_type]
    fusion_loss = fusion_loss()
    return fusion_loss

