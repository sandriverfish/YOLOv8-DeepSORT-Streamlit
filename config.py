#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]

# Classes
CLASSES = [0]
# 0-"person",

# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

CLASSIFICATION_MODEL_DIR = ROOT / 'weights' / 'classification'
YOLOv8n_cls = CLASSIFICATION_MODEL_DIR / "yolov8n-cls.pt"
YOLOv8s_cls = CLASSIFICATION_MODEL_DIR / "yolov8s-cls.pt"
YOLOv8m_cls = CLASSIFICATION_MODEL_DIR / "yolov8m-cls.pt"
YOLOv8l_cls = CLASSIFICATION_MODEL_DIR / "yolov8l-cls.pt"
YOLOv8x_cls = CLASSIFICATION_MODEL_DIR / "yolov8x-cls.pt"

CLASSIFICATION_MODEL_LIST = [
    "yolov8n-cls.pt",
    "yolov8s-cls.pt",
    "yolov8m-cls.pt",
    "yolov8l-cls.pt",
    "yolov8x-cls.pt",
]

SEGMENTATION_MODEL_DIR = ROOT / 'weights' / 'segmentation'
YOLOv8n_seg = SEGMENTATION_MODEL_DIR / "yolov8n-seg.pt"
YOLOv8s_seg = SEGMENTATION_MODEL_DIR / "yolov8s-seg.pt"
YOLOv8m_seg = SEGMENTATION_MODEL_DIR / "yolov8m-seg.pt"
YOLOv8l_seg = SEGMENTATION_MODEL_DIR / "yolov8l-seg.pt"
YOLOv8x_seg = SEGMENTATION_MODEL_DIR / "yolov8x-seg.pt"

SEGMENTATION_MODEL_LIST = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]

POSE_MODEL_DIR = ROOT / 'weights' / 'pose'
YOLOv8n_pose = POSE_MODEL_DIR / "yolov8n-pose.pt"
YOLOv8s_pose = POSE_MODEL_DIR / "yolov8s-pose.pt"
YOLOv8m_pose = POSE_MODEL_DIR / "yolov8m-pose.pt"
YOLOv8l_pose = POSE_MODEL_DIR / "yolov8l-pose.pt"
YOLOv8x_pose = POSE_MODEL_DIR / "yolov8x-pose.pt"
YOLOv8x_pose_p6 = POSE_MODEL_DIR / "yolov8x-pose-p6.pt"

POSE_MODEL_LIST = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
    "yolov8x-pose-p6.pt",
]

OBJECT_COUNTER = None
OBJECT_COUNTER1 = None