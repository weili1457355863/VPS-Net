from __future__ import division
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import copy
from shapely.geometry import Polygon

# compute the other two points from paired marking points
def compute_two_points(degree, point1, point2):
    p1_p2 = point2 - point1
    p1_p2_norm = np.sqrt(p1_p2[0] ** 2 + p1_p2[1] ** 2)
    # print('The distance between point1 and point2:{}'.format(p1_p2_norm))
    if p1_p2_norm < 200:
        depth = 250
    else:
        depth = 125
    p1_p2_unit = p1_p2 / p1_p2_norm
    rotate_matrix = np.array([[np.cos(degree / 180 * np.pi), np.sin(degree / 180 * np.pi)],
                              [-np.sin(degree / 180 * np.pi), np.cos(degree / 180 * np.pi)]])
    p2_p3 = np.dot(rotate_matrix, p1_p2_unit) * depth
    point3 = point2 + p2_p3
    point4 = point1 + p2_p3
    return point3, point4

# determine the final four points of the parking slot.
def compute_four_points(angle, point1, point2):
    point3, point4 = compute_two_points(angle, point1, point2)
    bbx_ps = np.concatenate((point1, point2, point3, point4), axis=0).reshape((4,2))
    point_12_min_x = np.min(bbx_ps[:2, 0])
    point_12_min_y = np.min(bbx_ps[:2, 1])
    point_12_max_x = np.max(bbx_ps[:2, 0])
    point_12_max_y = np.max(bbx_ps[:2, 1])
    point1_ex = copy.copy(point2)
    point2_ex = copy.copy(point1)
    point3_ex, point4_ex = compute_two_points(angle, point1_ex, point2_ex)
    bbx_ps_ex = np.concatenate((point1_ex, point2_ex, point3_ex, point4_ex), axis=0).reshape((4,2))
    bbx_car = np.array([[200,130], [400, 130], [400, 440], [200, 440]]) # 稍微进行了放大, 实际车位的坐标为[250,180,350,390]
    bbx_car_real = np.array([[250, 180], [350, 180], [350, 390], [250, 390]])
    iou_value = Polygon(bbx_ps).intersection(Polygon(bbx_car)).area
    iou_value_ex = Polygon(bbx_ps_ex).intersection(Polygon(bbx_car)).area
    iou_value_real = Polygon(bbx_ps).intersection(Polygon(bbx_car_real)).area
    iou_value_ex_real = Polygon(bbx_ps_ex).intersection(Polygon(bbx_car_real)).area
    # the vehicle is around the parking slot
    if iou_value < iou_value_ex:
        pts = np.array([point1, point2, point3, point4], np.int32)

    else:
        pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)

    diff_y = abs(point1[1] - point2[1])

    # the vehicle is in the parking slot
    if (diff_y < 70 and point_12_min_y > 300) or diff_y< 30 :
        if 180 < point_12_max_y < 390 or 180 < point_12_min_y < 390 or diff_y < 10:
            if point3[1] < 300 or point4[1] <300 :
                pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)
            else:
                pts = np.array([point1, point2, point3, point4], np.int32)
        if iou_value_real == 21000:
            print('The whole car is in the verticl parking slot')
            pts = np.array([point1, point2, point3, point4], np.int32)
        if iou_value_ex_real == 21000:
            print('The whole car is in the verticl parking slot')
            pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)

    # The vehile is in the parallel parking slot
    rec_1 = np.array([250, 180])
    rec_2 = np.array([350, 390])
    rec_3 = np.array([350, 180])
    rec_4 = np.array([250, 390])
    label_inter_1 = segment(point1, point2, rec_1, rec_2)
    label_inter_2 = segment(point1, point2, rec_3, rec_4)
    if diff_y > 180:

        if label_inter_1 or label_inter_2:
            if point1[1] < point2[1]:
                if point1[0] > point2[0]:
                    pts = np.array([point1, point2, point3, point4], np.int32)
                else:
                    pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)
            else:
                if point1_ex[0] > point2_ex[0]:
                    pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)
                else:
                    pts = np.array([point1, point2, point3, point4], np.int32)
        if iou_value_real ==  21000 or iou_value_ex_real == 21000:
            if point_12_min_x > 300:
                if point3[0] < 300 or point4[0] < 300:
                    pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)
                else:
                    pts = np.array([point1, point2, point3, point4], np.int32)
            if point_12_max_x < 300:
                if point3[0] > 300 or point4[0] > 300:
                    pts = np.array([point1_ex, point2_ex, point3_ex, point4_ex], np.int32)
                else:
                    pts = np.array([point1, point2, point3, point4], np.int32)
    return pts

def cross(p1,p2,p3):
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1

# determine whether the two line segments intersect
def segment(p1,p2,p3,p4):

    if(max(p1[0],p2[0])>=min(p3[0],p4[0])
    and max(p3[0],p4[0])>=min(p1[0],p2[0])
    and max(p1[1],p2[1])>=min(p3[1],p4[1])
    and max(p3[1],p4[1])>=min(p1[1],p2[1])):
      if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
        and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
        D=1
      else:
        D=0
    else:
      D=0
    return D
def to_cpu(tensor):
    return tensor.detach().cpu()

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)  左上和右下顶点
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            if int(detections[0, -1]) == 3:  # 针对第四类points需要考虑在同类间做最大值抑制
                label_match = detections[0, -1] == detections[:, -1] # 去掉非极大值抑制需要同类的条件
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                invalid = large_overlap & label_match  # 去掉非极大值抑制需要同类的条件
            else:
                invalid = large_overlap
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

# Mask the image according to the crop region
def fixed_ROI(img, points):
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array(points, np.int32)  # 顶点集
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
    return ROI


# Determine the crop region according to four vertices
def crop_margin(point1, point2, point3, point4):
    if point3[0] < 0:
        k = (point3[1] - point2[1]) / (point3[0] - point2[0])
        point3[0] = 0
        point3[1] = point2[1] - point2[0] * k
    if point4[0] < 0:
        k = (point4[1] - point1[1]) / (point4[0] - point1[0])
        point4[0] = 0
        point4[1] = point1[1] - point1[0] * k
    if point3[0] > 599:
        k = (point3[1] - point2[1]) / (point3[0] - point2[0])
        point3[0] = 599
        point3[1] = point2[1] + (599 - point2[0]) * k
    if point4[0] > 599:
        k = (point4[1] - point1[1]) / (point4[0] - point1[0])
        point4[0] = 599
        point4[1] = point1[1] + (599 - point1[0]) * k
    if point3[1] < 0:
        k = (point3[1] - point2[1]) / (point3[0] - point2[0])
        point3[1] = 0
        point3[0] = point2[0] - point2[1] / k
    if point4[1] < 0:
        k = (point4[1] - point1[1]) / (point4[0] - point1[0])
        point4[1] = 0
        point4[0] = point1[0] - point1[1] / k
    if point3[1] > 599:
        k = (point3[1] - point2[1]) / (point3[0] - point2[0])
        point3[1] = 599
        point3[0] = point2[0] + (599 - point2[1]) / k
    if point4[1] > 599:
        k = (point4[1] - point1[1]) / (point4[0] - point1[0])
        point4[1] = 599
        point4[0] = point1[0] + (599 - point1[1]) / k
    points_roi = np.vstack((point1, point2, point3, point4))
    points_roi = np.array(points_roi, np.float32)
    return points_roi

# a simple timer
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
