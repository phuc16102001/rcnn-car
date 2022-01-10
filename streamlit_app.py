import numpy as np
import streamlit as st
from PIL import Image

from car_detector import preds, get_ss

def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    return np.array(img)

title = """
<h1 style='text-align: center;'>CAR DETECTOR USING R-CNN</h1>
"""

def create_ui():
    st.markdown(title, unsafe_allow_html=True)
    
    image_file = st.file_uploader("Choose Image to detect", type=["png", "jpg", "jpeg"])
    threshold = st.slider("Select the threshold", min_value=0.5, max_value=1.0, value=0.7)
    ss_mode = st.radio("Selective search mode",("Single","Fast","Quality"))
    detect_button = st.button("Detect")
    
    col1, col2, col3 = st.columns([1, 6, 1])
    st.write("Threshold = " + str(threshold))
    
    with col1:
        pass
    with col2:
        if image_file is not None:
            img = load_image(image_file)
            col1.image(img, width=250,use_column_width=True)
    with col3:
        pass
    
    if detect_button:
        if image_file is not None:
            st.write("Waiting for selective search...")
            
            rects = get_ss(img,ss_mode)
            st.write("Selective search: "+str(len(rects)))
            st.write("Waiting for classification...")
            
            progress_bar = st.progress(0)
            
            img_predicted = preds(img, threshold, rects, callback=lambda x:progress_bar.progress(x))
            st.image(img_predicted, width=250, use_column_width=True)
        else:
            st.write("You have to choose image to detect.")

if __name__ == "__main__":
    create_ui()