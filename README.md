# VPS-Net
A vacant parking slot detection method in the around view image based on deep learning.

## Requirement
We ran our experiments with PyTorch 1.0.1, CUDA 9.0, Conda with Python 3.6 and Ubuntu 16.04.
## Installation
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd VPS-Net
    $ conda create --name vps-net python=3.6
    $ conda activate vps-net
    $ pip install -r requirements.txt

##### Download pretrained weights
    $ mkdir weights
    $ cd weights/
Download the [parking slot detection weights]() and the [paking slot occcupancy classification weights]() 

##### Download ps2.0 and PSV dataset
    $ mkdir data
    $ cd data/
Download the [ps2.0 dataset]() or the [PSV dataset]() 

## Test
Uses pretrained weights to detect the vacant parking slot in the around view image.
 
```
$ vps_net.py [-h] [--input_folder INPUT_FOLDER]
                  [--output_folder OUTPUT_FOLDER] [--model_def MODEL_DEF]
                  [--weights_path_yolo WEIGHTS_PATH_YOLO]
                  [--weights_path_vps WEIGHTS_PATH_VPS]
                  [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                  [--img_size IMG_SIZE] [--save_files SAVE_FILES]

```

#### Example (ps2.0 dataset)
Test on the ps2.0 dataset. The detection results including images and files will be saved.
```
$ python vps_net.py --input_folder data/ps2.0/testing/all --save_files 1
```

#### Testing results
<p align="center"><img src="assets/results.png" width="480"\></p>

## Exral annotations
In order to facilitate other researchers, the [annoations]() for vacant parking slots of ps 2.0 and PSV datasets has been 
made publicly avaliable.
