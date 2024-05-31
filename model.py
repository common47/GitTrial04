import streamlit as st
import cv2
import torch
import numpy as np
# Import Essential Libraries
import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pathlib
import glob
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings('ignore')

# YOLO 모델 불러오기
def load_yolo_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Darknet("yolov3.cfg")
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.to(device).eval()
    return model, device

# 객체 감지 함수
def detect_objects(image, model, device, conf_threshold=0.5, nms_threshold=0.4):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        detections = model(img_tensor)
        detections = non_max_suppression(detections, conf_threshold, nms_threshold)[0]
    return detections

# Streamlit 앱 UI
st.title("YOLO Object Detection")

# YOLO 모델 경로
model_path = r"C:\Users\ime203\Desktop\Graduation\runs\detect\Epochs80test\weights\best.pt"

uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[0]
    if file_type == "image":
        image = cv2.imread(uploaded_file.name)
        model, device = load_yolo_model(model_path)
        detections = detect_objects(image, model, device)
        # 여기서 감지된 객체를 시각화하거나 다른 처리를 수행할 수 있습니다.
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    elif file_type == "video":
        st.video(uploaded_file)
        # 비디오 처리 로직을 여기에 추가할 수 있습니다.
