# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:49:25 2025

@author: HP
"""

import streamlit as st
import numpy as np
import cv2
import pickle

# Load the pretrained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, 1)

    # Show the image
    st.image(input_img, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_img = cv2.resize(input_img, (224, 224))
    input_img = input_img / 255.0
    input_img = np.reshape(input_img, (1, 224, 224, 3))

    # Make prediction
    input_prediction = model.predict(input_img)
    input_label = np.argmax(input_prediction)

    # Show result
    if input_label == 1:
        st.success("Dog")
    else:
        st.success("Cat")
