#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  description: Parking slot detection
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from models.Yolov3 import Darknet
from utils.utils import non_max_suppression, rescale_boxes, pad_to_square, resize

class PsDetect(object):
    """
    Return paired marking points and angle
    """
    def __init__(self, model_def, model_path, img_size, device):
        self.model_yolov3 = Darknet(model_def, img_size=img_size).to(device)
        self.model_yolov3.load_state_dict(torch.load(model_path))
        self.model_yolov3.eval()
        self.device = device
        self.img_size = img_size

    # Detect the head of the parking slot and marking points
    def detect_object(self, img, conf_thres, nms_thres):
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        detection = self.model_yolov3(img)
        # From [x,y,w,h] to [x_l,y_l,x_r,y_r]
        detection = non_max_suppression(detection, conf_thres,
                                        nms_thres)
        if detection[0] is not None:
            detection = rescale_boxes(detection[0], self.img_size, (600, 600))
        return detection
    # The points in the head of the parking slot is less than 2
    def from_head_points(self, bbx, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1_l = bbx[0] + 24
        y1_l = bbx[1] + 22
        x2_l = bbx[2] - 24
        y2_l = bbx[3] - 22
        x1_r = bbx[2] - 24
        y1_r = bbx[1] + 22
        x2_r = bbx[0] + 24
        y2_r = bbx[3] - 22
        k_l = (y2_l - y1_l) / (x2_l - x1_l)
        k_r = (y1_r - y2_r) / (x1_r - x2_r)
        sum_intensity_l = 0
        sum_intensity_r = 0
        for i in range(int(x2_l - x1_l)):
            for k in range(-2, 2):
                y = int(k_l * i + y1_l + k)
                x = int(i + x1_l)
                if y > 599:
                    y = 599
                if x > 599:
                    x = 599
                if y < 0:
                    y = 0
                if x < 0:
                    x = 0
                sum_intensity_l += gray_img[y, x]
        for i in range(int(x1_r - x2_r)):
            for k in range(-2, 2):
                y = int(k_r * i + y2_r + k)
                x = int(i + x2_r)
                if y > 599:
                    y = 599
                if x > 599:
                    x = 599
                if y < 0:
                    y = 0
                if x < 0:
                    x = 0
                sum_intensity_r += gray_img[y, x]
        if sum_intensity_l > sum_intensity_r:
            if y2_l > y1_l:
                point1_x = x1_l
                point1_y = y1_l
                point2_x = x2_l
                point2_y = y2_l
            else:
                point1_x = x2_l
                point1_y = y2_l
                point2_x = x1_l
                point2_y = y1_l
        else:
            if y2_r > y1_r:
                point1_x = x1_r
                point1_y = y1_r
                point2_x = x2_r
                point2_y = y2_r
            else:
                point1_x = x2_r
                point1_y = y2_r
                point2_x = x1_r
                point2_y = y1_r
        point1 = np.array([point1_x, point1_y])
        point2 = np.array([point2_x, point2_y])
        return point1, point2
    # Parking slot detection
    def detect_ps(self, img, conf_thres, nms_thres):
        detection = self.detect_object(img, conf_thres, nms_thres)
        num_boxes = 0
        points_x = []
        points_y =[]
        ps = []
        if detection[0] is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                if int(cls_pred) == 3:
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    points_x.append(x)
                    points_y.append(y)
                else:
                    num_boxes += 1
            if num_boxes > 0:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    points_valid_x = []
                    points_valid_y = []
                    if int(cls_pred != 3):
                        for x, y in zip(points_x, points_y):
                            if x > x1 and x < x2 and y > y1 and y < y2:
                                points_valid_x.append(x)
                                points_valid_y.append(y)
                        if len(points_valid_x) == 2:
                            point1 = np.array((points_valid_x[0].item(), points_valid_y[0].item()))
                            point2 = np.array((points_valid_x[1].item(), points_valid_y[1].item()))
                        else:
                            bbx = [x1, y1, x2, y2]
                            point1, point2 = self.from_head_points(bbx, img)
                        if int(cls_pred) == 0:
                            angle = 90
                        elif int(cls_pred) == 1:
                            angle = 67
                        else:
                            angle = 129
                        ps.append([point1, point2, angle])
        return ps


