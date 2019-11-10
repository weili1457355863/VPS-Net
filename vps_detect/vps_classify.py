#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  description: Parking slot occupancy classification
"""
import torch
from torchvision import transforms
import numpy as np
import cv2

from utils.utils import crop_margin, fixed_ROI
from models.Customized import CustomizedAlexNet

class vpsClassify(object):
    """
    Return whether the paking slot is vacant. 0: vacant 1: non-vancant
    """
    def __init__(self, model_path, device):
        self.model_customized = CustomizedAlexNet()
        self.model_customized.eval()
        self.model_customized.load(model_path)
        self.model_customized.to(device)
        self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        self.device = device

    # parking slot regularization
    def image_preprocess(self, img, pts):
        points_roi = crop_margin(pts[0], pts[1], pts[2], pts[3])
        roi_img = fixed_ROI(img, points_roi)
        crop_x_min = np.min(points_roi[:, 0]) + 1
        crop_x_max = np.max(points_roi[:, 0])
        crop_y_min = np.min(points_roi[:, 1]) + 1
        crop_y_max = np.max(points_roi[:, 1])
        if pts[1][1] > pts[0][1]:
            points_dst = np.array([[crop_x_max, crop_y_min],
                                   [crop_x_max, crop_y_max],
                                   [crop_x_min, crop_y_max],
                                   [crop_x_min, crop_y_min]], np.float32)
        else:
            points_dst = np.array([[crop_x_max, crop_y_min],
                                  [crop_x_max, crop_y_max],
                                  [crop_x_min, crop_y_max],
                                  [crop_x_min, crop_y_min]], np.float32)
        m_warp = cv2.getPerspectiveTransform(points_roi, points_dst)
        warp_img = cv2.warpPerspective(roi_img, m_warp, (600, 600))
        crop_img = warp_img[int(crop_y_min):int(crop_y_max), int(crop_x_min):int(crop_x_max)]
        if (crop_img.shape[0] / crop_img.shape[1]) > 2:
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
        regul_img = cv2.resize(crop_img, (120, 46))
        return regul_img



    # parking slot occupancy classification
    def vps_classify(self,img, pts):
        regul_img = self.image_preprocess(img, pts)
        regul_img = self.transform(regul_img)
        regul_img = regul_img.to(self.device)
        regul_img = regul_img.unsqueeze(0)
        output = self.model_customized(regul_img)
        _, pred = torch.max(output.data, 1)
        return pred.item()



