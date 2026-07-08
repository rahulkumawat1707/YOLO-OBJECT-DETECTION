# YOLO Object Detection Project
CODTECH-INTERN ID: CTTS094
Name: Rahul Kumawat
Project Name: YOLO Object Detection Project
Project Scope: This project focuses on training a YOLOv8 object detection model on a custom campus image dataset and testing it on validation images.


This is my object detection project using YOLOv8. I made this project to train a custom model on images collected around the campus area and detect different objects like trees, flowers, cars, humans, buildings, roads, and some other outdoor objects.

The project includes the dataset, labels, training file, testing file, and the YOLO data configuration file. I have kept the folder simple so it can be opened and checked easily.

## Folder structure

```text
.
|-- dataset/
|   |-- images/
|   |   |-- train/
|   |   `-- val/
|   `-- labels/
|       |-- train/
|       `-- val/
|-- json_labels/
|-- labels/
|-- sample_predictions/
|-- convert.py
|-- data.yaml
|-- train.py
|-- test.py
|-- models/
|   `-- best.pt
|-- requirements.txt
|-- yolo26n.pt
`-- yolov8n.pt
```

## What each file is for

`data.yaml` has the dataset paths and class names used by YOLO.

`train.py` is used to train the YOLO model on the custom dataset.

`test.py` is used to validate the trained model and run prediction on one sample image.

`models/best.pt` is the trained model file saved after retraining.

`convert.py` was used to convert the JSON annotation files into YOLO text label format.

`dataset/` contains the final train and validation images with their matching YOLO labels.

`json_labels/` contains the original annotation files.

`labels/` contains the converted YOLO label files before they were arranged into train and validation folders.

`sample_predictions/` contains a few prediction images saved after testing the trained model.

## Classes used

There are 35 classes in this project. Some of them are tree, yellow flower, car, humans, building, road, sky, palm tree, grass, orange flower, and ground.

## How to run

First install the required packages:

```bash
pip install -r requirements.txt
```

Then train the model:

```bash
python train.py
```

After training, test the model:

```bash
python test.py
```

The trained weights will be saved inside the `runs/` folder after training. I have also kept the final trained weight in `models/best.pt`, so testing can be done directly without searching inside the training output folders.

## Current note

While checking the project, I found that the labels needed to be placed inside `dataset/labels/train` and `dataset/labels/val` for YOLO to read them correctly. That has been fixed now. The dataset is loading properly and the training script is able to start training with real label data.

After fixing this, I retrained the model again and kept the final weight file in `models/best.pt`.

## Current result

After retraining, the model is able to predict objects on validation images. The result is still not perfect because the dataset is small and has many classes, but the full training and prediction pipeline is working.

The last validation result I got was:

```text
mAP50: 0.0959
mAP50-95: 0.0504
precision: 0.6593
recall: 0.0816
```

Predicted images are generated inside the `runs/detect/check_predict` folder when `test.py` or prediction commands are run. I have not kept the full `runs/` folder for GitHub because it contains generated output files, but I added a few selected outputs inside `sample_predictions/`.

## Dataset split

The dataset is arranged like this:

```text
Training images: 160
Training labels: 160
Validation images: 39
Validation labels: 39
```

## My understanding

This project helped me understand how object detection datasets are prepared, how labels are matched with images, and how YOLO trains on a custom dataset. The main issue I faced was label placement, because YOLO expects labels to follow the same train and validation split as the images. Once the labels were arranged correctly, the training pipeline started working properly.
