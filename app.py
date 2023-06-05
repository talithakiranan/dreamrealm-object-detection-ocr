import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import pytesseract
import numpy as np
import pandas as pd
import seaborn as sn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from PIL import Image
import pickle
import streamlit as st
from streamlit_option_menu import option_menu as option
import base64

pg_title = "Dream Realm Project"
icon = "Dream_Realm.png"

st.set_page_config(page_title=pg_title, page_icon=icon, layout="wide")

st.sidebar.image(icon, caption="Logo Dream Realm", width=235)

# Add background image
def bg_img():
    st.markdown(
        f"""
        <style>
         .stApp {{
            background-image: url("https://img.freepik.com/free-photo/pink-sky-background-with-crescent-moon_53876-129048.jpg");
            background-position: center; /*Center the image */
            background-repeat: no-repeat;
            background-size: cover;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0)
        }}
        [data-testid="stToolbar"] {{        
        right: 2rem;
        }}
        [data-testid="stSidebar"] {{        
        background-image: url("https://img.freepik.com/free-photo/blue-sky-with-clouds-background_53876-128611.jpg");
        background-repeat: no-repeat;
        background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True)
bg_img()

# Membuat Navigasi Sidebar
with st.sidebar:
    selected = option("Main Menu", ["Home", 'Project'],
                     icons=['house', 'gear', 'code-square'], menu_icon="list", default_index=0)

if selected == 'Home':
    st.write('<h1 style="text-align: center;">Welcome to Dream Realm</h1>', unsafe_allow_html=True)
    st.subheader('''
    Dream Realm adalah team yang memiliki sebuah project Computer Vision dengan lingkup Object Detection dan OCR.''')
    st.subheader(
        "Pada project ini pengguna dapat melakukan *detection dan text extraction* pada bukti transfer digital (mobile banking) berupa nominal, pengirim, penerima, dan tanggal.")
    st.markdown('### ')
    st.info('''
    - If you have questions, get in touch with us: dreamrealm369@gmail.com''')
    st.markdown('''
    Let's Get Started!
     > 👈🏻 Silakan pilih menu *'Project'* di samping untuk melakukan demo project.
    ''')

video_urls = {'Video 1': 'https://www.youtube.com/watch?v=UHX6zmMUShk',
              'Video 2': 'https://www.youtube.com/watch?v=7y4WhR3W2Nc',
              'Video 3': 'https://www.youtube.com/watch?v=AclJEg7psPw'}
vid_selection = st.sidebar.selectbox('Pilih video', list(video_urls.keys()))  # dropdown unt memilih video
st.sidebar.video(video_urls[vid_selection])  # putar video pilihan

# Define the class labels
if selected == 'Project':
    class_labels = ['Nominal', 'Pengirim', 'Penerima', 'Tanggal']

    # Load the trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    num_classes = 5  # background + 4 class labels
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the image transformation
    transform = T.Compose([
        T.ToTensor()
    ])

    # Function to perform object detection and OCR
    def detect_and_extract(image):
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)

        boxes = prediction[0]['boxes'].to('cpu').numpy()
        labels = prediction[0]['labels'].to('cpu').numpy()

        detections = []
        for box, label in zip(boxes, labels):
            if label > 0:  # Exclude the background label
                xmin, ymin, xmax, ymax = box.astype(np.int)
                object_image = image.crop((xmin, ymin, xmax, ymax))

                # Perform OCR on the object image
                object_image_gray = object_image.convert('L')
                object_text = pytesseract.image_to_string(object_image_gray)

                detections.append((class_labels[label-1], object_text))

        return detections

    # Function to display the detections
    def display_detections(image, detections):
        fig, ax = plt.subplots()
        ax.imshow(image)

        for label, text in detections:
            ax.text(10, 10, f'{label}: {text}', bbox=dict(facecolor='white', alpha=0.7))

        ax.axis('off')
        st.pyplot(fig)

    st.title("Object Detection and OCR on Bank Transfer Invoices")

    uploaded_image = st.file_uploader('Upload your bank transfer invoice here', type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform object detection and OCR
        detections = detect_and_extract(image)

        # Display the detections
        display_detections(image, detections)

# Add custom footer
def footer():
    st.markdown(
        """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        </style>
        """
        , unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        </style>
        """
        , unsafe_allow_html=True
    )
footer()
