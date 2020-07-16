# VPS-Net
A vacant parking slot detection method in the around view image based on deep learning.

## Requirement
We ran our experiments with PyTorch 1.0.1, CUDA 9.0, Conda with Python 3.6 and Ubuntu 16.04.
## Installation
##### Clone and install requirements
    $ git clone https://github.com/weili1457355863/VPS-Net.git
    $ cd VPS-Net
    $ conda create --name vps-net python=3.6
    $ conda activate vps-net
    $ pip install -r requirements.txt

##### Download pretrained weights
    $ mkdir weights
    $ cd weights/
Download the [weights](https://drive.google.com/file/d/1mkrQ5ehgZY5iOM3HnXR5hPBw1kjXXo6a/view?usp=sharing) of detection network and classification network.

##### Download ps2.0 and PSV dataset
    $ mkdir data
    $ cd data/
Download the [ps2.0 dataset](https://cslinzhang.github.io/deepps/) or the [PSV dataset](http://cs1.tongji.edu.cn/tiev/resourse/) 

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

## Extral annotation
In order to facilitate other researchers, the [annoation](https://drive.google.com/file/d/1IQxiXrfdIxfpHaTWyXGHQCvZUSRMTGSK/view?usp=sharing) for vacant parking slots of ps 2.0 and PSV datasets has been 
made publicly avaliable.

## Citation
```
@article{li_vacant_2020,
	title = {Vacant Parking Slot Detection in the Around View Image Based on Deep Learning},
	volume = {20},
	doi = {10.3390/s20072138},
	pages = {2138},
	journal = {Sensors},
	author = {Li, Wei and Cao, Libo and Yan, Lingbo and Li, Chaohui and Feng, Xiexing and Zhao, Peijie},
	date = {2020-04-10}
}

```


