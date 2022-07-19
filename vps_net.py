#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  description:VPS_Net: Vacant parking slot detection
"""
from __future__ import division

import argparse
import torch
import os
import cv2
import numpy as np
import copy
import glob
import tqdm
from PIL import Image

from vps_detect.ps_detect import PsDetect
from vps_detect.vps_classify import vpsClassify
from utils.utils import compute_four_points

# add image types at will
img_types = ["jpg", 'bmp', 'png']


def get_images_from_folder(path):
    images = []
    [images.extand(glob.glob(os.path.join(path,"*." + img_type))) for img_type in img_types]
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="data/outdoor-rainy", help="path to dataset")
    parser.add_argument("--output_folder", type=str, default="output/outdoor-rainy", help="path to output")
    parser.add_argument("--model_def", type=str, default="config/ps-4.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path_yolo", type=str, default="weights/yolov3_4.pth",
                        help="path to yolo weights file")
    parser.add_argument("--weights_path_vps", type=str, default="weights/Customized.pth",
                        help="path to vps weights file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold") # 0.9
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--save_files", type=bool, default=False, help="save detected results")
    opt = parser.parse_args()
    
    opt.output_folder = os.path.normpath(opt.output_folder)

    os.makedirs(opt.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ps_detect =PsDetect(opt.model_def, opt.weights_path_yolo, opt.img_size, device)
    vps_classify = vpsClassify(opt.weights_path_vps, device)

    with torch.no_grad():
        imgs_list = get_images_from_folder(opt.input_folder)
        print("input_folder: " + opt.input_folder)
        print("number of images: " + str(len(imgs_list)))
        for img_path in tqdm.tqdm(imgs_list):
            if opt.save_files:
                img_name = os.path.split(img_path)[1]
                filename = img_name.split('.')[0] + '.txt'
                file_path = os.path.join(opt.output_folder, filename)
                file = open(file_path, 'w')
            img = np.array(Image.open(img_path))
            if img.shape[0] != 600:
                img = cv2.resize(img, (600, 600))
            detections = ps_detect.detect_ps(img, opt.conf_thres, opt.nms_thres)
            if len(detections) !=0:
                for detection in detections:
                    point1 = detection[0]
                    point2 = detection[1]
                    angle = detection[2]
                    pts = compute_four_points(angle, point1, point2)
                    point3_org = copy.copy(pts[2])
                    point4_org = copy.copy(pts[3])
                    label_vacant = vps_classify.vps_classify(img, pts)
                    if label_vacant == 0:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    pts_show = np.array([pts[0], pts[1], point3_org, point4_org], np.int32)
                    if opt.save_files:
                        file.write(str(angle))
                        file.write(' ')
                        points = list((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
                        for value in points:
                            file.write(str(value.item()))
                            file.write(' ')
                        file.write('\n')
                    cv2.polylines(img, [pts_show], True, color, 2)
            cv2.imshow('Detect PS', img[:,:,::-1])
            cv2.waitKey(1)
            if opt.save_files:
                file.close()
                cv2.imwrite(os.path.join(opt.output_folder, img_name), img[:,:,::-1])
        cv2.destroyAllWindows()






