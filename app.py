#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection", "Classification", "Segmentation", "Pose"]
)

model_type = None
model_path = ""

if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
elif task_type == "Classification":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.CLASSIFICATION_MODEL_LIST
    )
    model_path = Path(config.CLASSIFICATION_MODEL_DIR, str(model_type))
elif task_type == "Segmentation":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.SEGMENTATION_MODEL_LIST
    )
    model_path = Path(config.SEGMENTATION_MODEL_DIR, str(model_type))
elif task_type == "Pose":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.POSE_MODEL_LIST
    )
    model_path = Path(config.POSE_MODEL_DIR, str(model_type))
else:
    st.error("Currently only 'Detection/Classification/Segmentation/Pose' function are implemented")
    st.error("Please Select Model in Sidebar")


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

# define class to be processed in model
