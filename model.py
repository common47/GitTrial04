import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# YOLO 모델 로드
model_path = r"C:\Users\ime203\Desktop\Graduation\runs\detect\Epochs80test\weights\best.pt"  # 훈련된 모델 파일의 경로
model = YOLO(model_path)

# Streamlit 앱 UI
st.title("YOLO Object Detection")

# 이미지 또는 비디오 업로드
uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[0]
    if file_type == "image":
        # 이미지 파일인 경우
        image = cv2.imread(uploaded_file.name)
        results = model(image)  # 이미지에서 객체 감지
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    elif file_type == "video":
        # 비디오 파일인 경우
        video = cv2.VideoCapture(uploaded_file.name)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            results = model(frame)  # 비디오에서 객체 감지
            # 여기서 결과를 사용하여 프레임에 객체를 그릴 수 있습니다.
            # 예: results.show()
            st.image(frame, channels="BGR", use_column_width=True)
else:
    st.write("Please upload an image or video file.")
