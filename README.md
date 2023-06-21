
# Vehicle Detection using YOLOv5

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-v5.0%2B-orange)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

This repository contains the code and resources for training a vehicle detection model using YOLOv5 and a custom dataset. The goal of this project is to detect and localize vehicles in images or videos, enabling various applications such as traffic monitoring, object tracking, and autonomous driving.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

These instructions will help you set up the project and train your own vehicle detection model.

### Prerequisites

- Python 3.8 or above
- NVIDIA GPU (recommended) with CUDA support
- PyTorch 1.7 or above
- OpenCV
## Dataset Preparation

To train a vehicle detection model, you need a labeled dataset containing images or videos with annotated bounding boxes around the vehicles. Follow these steps to prepare your dataset:

1. Organize your dataset directory structure as follows:

dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...


Each image should have a corresponding label file in the "labels" directory, where each line represents a bounding box annotation in the YOLO format `(class_index, x_center, y_center, width, height)`. The coordinates should be normalized between 0 and 1.

2. Update the `data.yaml` file:

- Set the number of classes in `nc` (including the background class if applicable).
- Specify the path to your training and validation data in `train` and `val` respectively.
- Customize other settings as per your requirements.
## Training

Start training your model with the following command:

```
python train.py --data data.yaml --cfg models/yolov5s.yaml --batch-size 16 --epochs 50
```

Adjust the batch size, number of epochs, and the model configuration file (yolov5s.yaml, yolov5m.yaml, yolov5l.yaml, or yolov5x.yaml) based on your resources and desired accuracy.

Monitor the training progress through the generated logs and checkpoints saved in the runs/ directory.

## Inference
Once you have a trained model, you can use it for vehicle detection on new images or videos.

For image inference, run:

```
python detect.py --source path/to/image.jpg --weights path/to/best.pt --conf 0.5
```
For video inference, run:

```
python detect.py --source path/to/video.mp4 --weights path/to/best.pt --conf 0.5 --output path/to/output.mp4
```
Set the confidence threshold (--conf) based on your desired precision-recall trade-off.


## Evaluation
To evaluate the performance of your trained model, you can use the provided test.py script:

```
python test.py --data data.yaml --weights path/to/best.pt --iou-thres 0.5
```
Adjust the IoU threshold (--iou-thres) based on your evaluation requirements.

## Contribution

<table>
<tr align="center">


<td>

Lakshmi Mani Shankar

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/91583687?s=400&u=0b12e9a254a9f85deb2ba2647eeac55ff4ca0f48&v=4"  height="120" >
</p>
<p align="center">
<a href = "https://github.com/Manishankar9977"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>

</p>
</td>

<td>

Krishna Pradeep

<p align="center">
  
<img src = "https://avatars.githubusercontent.com/u/90108144?v=4"  height="120" >
</p>
<p align="center">
<a href = "https://github.com/Krishna-0311"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>

</p>
</td>

<td>

Sujith Kumar

<p align="center">
  
<img src = "https://avatars.githubusercontent.com/u/96331881?v=4"  height="120" >
</p>
<p align="center">
<a href = "https://github.com/Sujith6502"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>

</p>
</td>



</tr>
  </table>
