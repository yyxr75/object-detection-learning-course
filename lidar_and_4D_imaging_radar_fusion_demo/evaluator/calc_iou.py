# -- coding: utf-8 --
import numpy as np
from evaluator.intersect_iou import box_intersection_area, box2corners
from evaluator.min_enclosing_box import smallest_bounding_box
import torch
import matplotlib.pyplot as plt

def calc_iou(box1, box2):
    '''
    :param predicted box1:
        9 [x,y,p,w,h,angle,p,class:0/1]
    :param ground truth box2:
        6 [id,x,y,w,h,angle]
    :return:
    '''
    # import pdb;pdb.set_trace()
    box1 = np.asarray([box1[0],box1[1],box1[3],box1[4],box1[5]*np.pi/180])
    box1 = box1.astype(float)
    box2 = box2[1:].astype(float)
    # import pdb;pdb.set_trace()
    box2[-1] = box2[-1]*np.pi/180
    # 计算相交面积
    inter_area, corners = box_intersection_area(box1, box2)
    # 计算并面积
    area1 = box1[2]*box1[3]
    area2 = box2[2]*box2[3]
    # 计算iou
    u = area1 + area2 - inter_area
    # import pdb;pdb.set_trace()
    iou = inter_area / (u+1e-6)
    # import pdb;pdb.set_trace()
    return iou, u

def calc_giou(box1, box2):
    '''
    :param predicted box1:
        9 [x,y,p,w,h,angle,p,class:0/1]
    :param ground truth box2:
        6 [id,x,y,w,h,angle]
    :return:
    '''
    # 计算普通iou
    iou, u = calc_iou(box1, box2)
    # 计算最小外包面积
    box1 = np.asarray([box1[0],box1[1],box1[3],box1[4],box1[5]*np.pi/180])
    box2 = box2[1:]
    box2[-1] = box2[-1]*np.pi/180
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h = smallest_bounding_box(tensor1)
    area_c = w*h
    # 得到了giou
    giou = iou+(area_c-u)/area_c
    return giou

def calc_diou(box1, box2):
    '''
    :param predicted box1:
        9 [x,y,p,w,h,angle,p,class:0/1]
    :param ground truth box2:
        6 [id,x,y,w,h,angle]
    :return:
    '''
    # 计算普通iou
    iou, u = calc_iou(box1, box2)
    # 计算最小外包面积
    box1 = np.asarray([box1[0],box1[1],box1[3],box1[4],box1[5]*np.pi/180])
    box2 = box2[1:]
    box2[-1] = box2[-1]*np.pi/180
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    # import pdb;pdb.set_trace()
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h = smallest_bounding_box(tensor1)
    c2 = w*w+h*h
    # 计算中心点的欧式距离
    x_offset = box1[0] - box2[1]
    y_offset = box1[1] - box2[2]
    d2 = x_offset*x_offset + y_offset*y_offset
    # 得到diou
    diou_loss = 1-iou+d2/c2
    # import pdb;pdb.set_trace()
    return diou_loss

def calc_ciou(box1, box2):
    '''
    :param predicted box1:
        9 [x,y,p,w,h,angle,p,class:0/1]
    :param ground truth box2:
        6 [id,x,y,w,h,angle]
    :return:
    '''
    # 计算普通iou
    iou, u = calc_iou(box1, box2)
    # 计算最小外包面积
    box1 = box1[0,1,3,4,5]
    box2 = box2[1:]
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h = smallest_bounding_box(tensor1)
    c2 = w*w+h*h
    # 计算中心点的欧式距离
    x_offset = box1[0] - box2[1]
    y_offset = box1[1] - box2[2]
    d2 = x_offset*x_offset + y_offset*y_offset
    # 计算长宽比相似性
    w_gt, h_gt = box2[2], box2[3]
    w_pred, h_pred = box1[2], box1[3]
    arctan = np.arctan(w_gt/h_gt) - np.arctan(w_pred/h_pred)
    v = (4/np.pi*np.pi)*np.power((arctan), 2)
    s = 1 - iou
    alpha = v / (s + v)
    w_temp = 2*w_pred
    ar = (8 / (np.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    # 得到ciou_loss
    ciou_loss = iou - (u + alpha * ar)
    return ciou_loss


def main():
    box1 = np.array([[1.7700e+02, 7.2040e+01, 9.0880e-01, 1.2870e+01, 2.4773e+01, 6.9586e+00,
         9.9990e-01, 9.9990e-01, 9.9212e-05],
        [2.2500e+02, 2.9604e+02, 9.0574e-01, 1.2518e+01, 2.4439e+01, 2.5717e+00,
         9.9988e-01, 9.9988e-01, 1.1701e-04],
        [2.9700e+02, 2.4804e+02, 9.0114e-01, 1.2555e+01, 2.4835e+01, 7.1307e+01,
         9.9996e-01, 9.9996e-01, 4.4476e-05],
        [1.6100e+02, 2.1605e+02, 8.9741e-01, 1.4957e+01, 3.0535e+01, 5.7846e-01,
         9.9994e-01, 9.9994e-01, 5.7519e-05],
        [2.1700e+02, 2.4043e+01, 8.8714e-01, 1.3700e+01, 2.8652e+01, 8.7812e-01,
         9.9987e-01, 9.9987e-01, 1.3172e-04],
        [2.2500e+02, 1.6804e+02, 8.6291e-01, 1.3069e+01, 2.5783e+01, 2.1645e+00,
         9.9984e-01, 9.9984e-01, 1.6396e-04],
        [1.4500e+02, 2.6405e+02, 8.4927e-01, 1.7486e+01, 4.4134e+01, 1.7434e+02,
         9.9994e-01, 9.9994e-01, 6.4964e-05],
        [2.2500e+02, 1.9204e+02, 8.4409e-01, 1.3490e+01, 2.7524e+01, 1.6841e+00,
         9.9979e-01, 9.9979e-01, 2.0873e-04],
        [2.3300e+02, 2.1604e+02, 8.3694e-01, 1.3006e+01, 2.8435e+01, 1.7288e+02,
         9.9984e-01, 9.9984e-01, 1.5968e-04],
        [2.2500e+02, 9.6045e+01, 8.3135e-01, 1.4499e+01, 2.9652e+01, 5.7747e+00,
         9.9994e-01, 9.9994e-01, 5.5067e-05],
        [2.2500e+02, 1.3605e+02, 8.2342e-01, 1.4480e+01, 3.4636e+01, 1.7764e+02,
         9.9971e-01, 9.9971e-01, 2.8785e-04],
        [2.0100e+02, 9.6044e+01, 8.1463e-01, 1.3940e+01, 2.9163e+01, 1.7562e+02,
         9.9975e-01, 9.9975e-01, 2.4501e-04],
        [2.0100e+02, 3.2038e+01, 6.9737e-01, 1.2237e+01, 2.4166e+01, 4.7391e+01,
         9.9981e-01, 9.9981e-01, 1.8917e-04]])
    box2 = np.array([[ 1.00000000e+00,  1.78957080e+02,  7.17562100e+01,
         1.41704750e+01,  2.79044300e+01, -5.17086349e-24],
       [ 1.40000000e+01,  2.01258450e+02,  9.98516700e+01,
         1.44527050e+01,  3.08091950e+01, -2.50199526e-21],
       [ 2.00000000e+00,  1.56042895e+02,  2.17876800e+02,
         1.45319100e+01,  2.92467800e+01, -2.14453446e-23],
       [ 1.00000000e+01,  2.29906505e+02,  2.18657050e+02,
         1.29466550e+01,  2.69606700e+01, -2.49505513e-21],
       [ 3.00000000e+00,  2.26915970e+02,  1.93144980e+02,
         1.31178400e+01,  2.68737300e+01, -2.44241380e-21],
       [ 4.00000000e+00,  2.27058565e+02,  1.70530335e+02,
         1.31181800e+01,  2.58652250e+01, -2.48116555e-21],
       [ 5.00000000e+00,  2.25353385e+02,  1.35690110e+02,
         1.46723100e+01,  3.41488350e+01, -2.43403254e-21],
       [ 6.00000000e+00,  2.21824230e+02,  9.87716500e+01,
         1.41303600e+01,  2.97381350e+01,  6.10001808e-23],
       [ 1.10000000e+01,  2.26329390e+02,  2.95201100e+02,
         1.16299850e+01,  2.30600600e+01, -2.46436299e-21],
       [ 1.50000000e+01,  1.98005620e+02,  2.54394750e+01,
         1.34085250e+01,  2.97516500e+01, -2.49929306e-21],
       [ 8.00000000e+00,  2.14905495e+02,  2.33006150e+01,
         1.46312350e+01,  2.97191500e+01, -2.46326308e-21],
       [ 1.20000000e+01,  1.46669495e+02,  2.66007620e+02,
         1.75225250e+01,  4.41690600e+01, -8.39565472e-23],
       [ 1.30000000e+01,  2.97186175e+02,  2.46843875e+02,
         1.08773150e+01,  2.10598300e+01, -1.26151392e-21]])

    box1 = box1[2,:]
    box2 = box2[-1,:]
    # x,y,p,w,h,angle,p,class:0/1
    # box1 = np.array([0,0,0.09,2,6,0.0,0.9,0,1])
    # ------------------> x
    # |    ---- w
    # |    |  |
    # |    |  |
    # |    ----
    # |    h
    # v
    # y

    # box1 = np.array([1.60500000e+02, 2.45000000e+01, 7.66985714e-01, 1.60000000e+02,
    #        1.60000000e+02, 2.70000000e+02, 5.00000000e-01, 9.99401093e-01,
    #        5.98885643e-04])

    # box2 = np.array([1.30000000e+02, 3.00010300e+02, 8.97352950e+01, 1.21814500e+01,
    #        2.48233700e+01, 3.67019088e-13])

    # iou,_ = calc_iou(box1,box2)
    # diou_loss = calc_diou(box1,box2)
    # iou = 1-iou
    # print(iou)
    # print(diou_loss)

    # ------------------> x
    # |    ---- w
    # |    |  |
    # |    |  |
    # |    ----
    # |    h
    # v
    # y
    # x,y,p,w,h,angle,p,class:0/1
    box1 = np.array([0,0,0.09,2,6,0.0,0.9,0,1]) # angle = 0
    iou_arr = np.zeros((360))
    u_arr = np.zeros((360))
    diou_arr = np.zeros((360))
    for theta in range(360):
        # box2 = np.array([100,0.09476,0.28428,2+0.1341,6+0.4023,theta])
        box2 = np.array([100,0.0,0.0,2.0,6.0,theta])
        # import pdb;pdb.set_trace()
        iou, u = calc_iou(box1, box2)
        diou_loss = calc_diou(box1, box2)
        # iou = 1 - iou
        # print(iou)
        # print(diou_loss)
        iou_arr[theta] = 1-iou
        u_arr[theta] = u
        diou_arr[theta] = diou_loss
    plt.plot(iou_arr)
    # plt.plot(u_arr)
    plt.plot(diou_arr)
    plt.xlabel('angle')
    plt.legend(['iou loss', 'diou loss'])
    plt.title('d/iou loss with angle change')
    plt.show()


if __name__=='__main__':
    main()