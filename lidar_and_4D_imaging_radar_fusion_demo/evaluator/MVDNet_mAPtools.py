# -- coding: utf-8 --
'''
适合torch1.7_cu110版本的detectron2安装指令:
python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
'''
from pycocotools.coco import COCO
from collections import defaultdict
import copy
from detectron2.evaluation.rotated_coco_evaluation import RotatedCOCOeval
from pycocotools.cocoeval import Params
import numpy as np
from evaluator.visualize import showPredResult
import torch
import os
import json
import itertools
from detectron2.utils.logger import create_small_table
from tabulate import tabulate
import pdb
import matplotlib.pyplot as plt
from loguru import logger

def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
        "keypoints": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
    }[iou_type]

    if coco_eval is None:
        self._logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    results = {
        metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
        for idx, metric in enumerate(metrics)
    }
    self._logger.info(
        "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
    )
    if not np.isfinite(sum(results.values())):
        self._logger.info("Note that some metrics cannot be computed.")

    if class_names is None or len(class_names) <= 1:
        return results

    precisions = coco_eval.eval["precision"]
    assert len(class_names) == precisions.shape[2]

    results_per_category = []
    for idx, name in enumerate(class_names):
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap * 100)))

    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP"] * (N_COLS // 2),
        numalign="left",
    )
    self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

    results.update({"AP-" + name: ap for name, ap in results_per_category})
    return results


def get_dt_offline(timestamp, map_out_path):
    data_path = os.path.join(map_out_path, 'detection-results', str(timestamp))
    data_filename = data_path+'.txt'
    with open(data_filename) as f:
        lines = f.readlines()
        num_boxes = len(lines) - 1
        pred_boxes = np.zeros((num_boxes, 9))
        for i in range(len(lines)):
            if i == 0:
                continue
            x,y,p1,w,h,angle,p2,cls1,cls2 = lines[i].split()
            pred_box = np.array([float(x), float(y), float(p1), float(w), float(h), float(angle), float(p2), float(cls1), float(cls2)])
            pred_boxes[i-1, :] = (pred_box)
    return pred_boxes


def calc_mvdnet_mAP(model, test_lines, dataset, opts, map_out_path):
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

    coco_results = []
    cnt = 0
    gt_objNum = 0

    # 统计结果
    angle_pred = np.zeros(360,)
    width_pred = []
    height_pred = []
    x_pred = []
    y_pred = []
    box_num_pred = []

    angle_gt = np.zeros(360,)
    width_gt = []
    height_gt = []
    x_gt = []
    y_gt = []
    box_num_gt = []

    for annotation_line in test_lines:
        cnt += 1
        print(cnt)
        # if cnt > 10:
        #     break
        line = annotation_line.split()
        frame_num = int(line[0])
        timestamp = int(line[1])
        [pillars, lidarData, radar_data, gt_boxes, class_arr] = dataset.get_data(annotation_line)

        # --------------------------
        # 创建模型，输入test数据集数据进行推理，并保存json结果
        # --------------------------
        pillars = np.expand_dims(pillars, 0)
        radar_data = np.expand_dims(radar_data, 0)
        radar_data = np.expand_dims(radar_data, 0)
        radar_data = radar_data.astype(np.float32)
        # get model prediction result
        with torch.no_grad():
            pillars = torch.from_numpy(pillars).type(torch.FloatTensor).cuda()
            radar_data = torch.from_numpy(radar_data).type(torch.FloatTensor).cuda()
            gt_boxes = torch.from_numpy(gt_boxes).type(torch.FloatTensor).cuda()
            predictions = model(pillars, radar_data, opts)
            radar_data = torch.squeeze(radar_data).cpu().detach().numpy()
            pred_boxes = showPredResult(lidarData, radar_data, gt_boxes, predictions, 0.2, opts)
            plt.pause(0.0001)
            plt.clf()
        # ---------------------------
        # 推理太慢了，直接读结果吧
        # ---------------------------
        # pred_boxes = get_dt_offline(timestamp, map_out_path)
        # 构造pycocotools结构数据
        if pred_boxes.size == 0:
            continue
        image = dict()
        image["timestamp"] = timestamp
        image["id"] = frame_num
        coco_gt.dataset["images"].append(image)

        box_num_gt.append(gt_boxes.shape[0])
        for i in range(gt_boxes.shape[0]):
            gt_objNum += 1
            coco_ann = dict()
            coco_ann["area"] = gt_boxes[i, 3] * gt_boxes[i, 4]
            coco_ann["iscrowd"] = 0
            coco_ann["image_id"] = frame_num
            coco_ann["bbox"] = copy.deepcopy([gt_boxes[i, 1], gt_boxes[i, 2], gt_boxes[i, 3], gt_boxes[i, 4], gt_boxes[i, 5]])
            coco_ann["category_id"] = 1
            coco_ann["id"] = gt_objNum
            coco_gt.dataset["annotations"].append(coco_ann)
            # 统计结果
            angle_gt[int(gt_boxes[i,5])] += 1
            width_gt.append(gt_boxes[i, 3])
            height_gt.append(gt_boxes[i, 4])
            x_gt.append(gt_boxes[i, 1])
            y_gt.append(gt_boxes[i, 2])

        box_num_pred.append(pred_boxes.shape[0])
        for i in range(pred_boxes.shape[0]):
            # 统计结果
            angle_pred[int(pred_boxes[i,5])] += 1
            width_pred.append(pred_boxes[i, 3])
            height_pred.append(pred_boxes[i, 4])
            x_pred.append(pred_boxes[i, 0])
            y_pred.append(pred_boxes[i, 1])

            # mvdnet angle = (true angle + 180) (+- 360)
            new_angle = pred_boxes[i,5] + 180
            if new_angle > 180:
                new_angle -= 360
            elif new_angle < -180:
                new_angle += 360
            result = dict()
            result["image_id"] = frame_num
            result["category_id"] = 1
            result["bbox"] = copy.deepcopy([pred_boxes[i,0], pred_boxes[i,1], pred_boxes[i,3], pred_boxes[i,4], new_angle])
            result["score"] =pred_boxes[i,2]
            coco_results.append(result)

    coco_gt.createIndex()
    # 把最终结果的json文件存下来
    if map_out_path:
        file_path = os.path.join(map_out_path, "coco_instances_results.json")
        with open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

    # --------------------------
    # 调用coco api 进行mAP估计
    # --------------------------
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = RobotCarCOCOeval(coco_gt, coco_dt, iouType="bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # 这个函数的作用是如果有其他的类别，则算个总平均之类的
    # res = _derive_coco_results(
    #     coco_eval, "bbox", class_names=self._metadata.get("thing_classes")
    # )

    plt.figure(1)
    plt.subplot(221)
    plt.title("box number distribution")
    plt.plot(box_num_gt,'bo')
    plt.plot(box_num_pred,'ro')
    plt.subplot(222)
    plt.title("angle distribution")
    plt.plot(angle_gt,'bo')
    plt.plot(angle_pred,'ro')
    plt.subplot(223)
    plt.title("width distribution")
    plt.plot(width_gt, 'bo')
    plt.plot(width_pred, 'ro')
    plt.subplot(224)
    plt.title("height distribution")
    plt.plot(height_gt,'bo')
    plt.plot(height_pred,'ro')

    plt.figure(2)
    x_gt = np.asarray(x_gt)
    x_pred = np.asarray(x_pred)
    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)
    plt.title("position distribution")
    plt.scatter(x_gt, y_gt, c='b')
    plt.scatter(x_pred, y_pred, c='r')

    plt.show()


class RobotCarCOCOeval(RotatedCOCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        if not iouType:
            print('iouType not specified. use default iouType segm')

        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = RobotCarParams(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        self.use_ext = False
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def summarize(self):
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.65, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.8, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[0])
            stats[5] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[1])
            stats[6] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        assert iouType == 'bbox', 'only bbox evaluation is supported for RobotCar oxford_dataset!'
        summarize = _summarizeDets
        self.stats = summarize()

class RobotCarParams(Params):
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

