import os
import easyocr
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

st.set_page_config(page_title=pg_title, 
                   page_icon=icon,  layout="wide") 

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

if (selected == 'Home'):
    #email = "dreamrealm369@gmail.com"
    st.write('<h1 style="text-align: center; {color: blue;}">Welcome to Dream Realm</h1>', unsafe_allow_html=True)
    #st.title("_Object Detection and Text Extraction_")
    st.subheader('''
    Dream Realm adalah team yang memiliki sebuah project Computer Vision dengan lingkup _Object Detection dan OCR_.''') 
    st.subheader("Pada project ini pengguna dapat melakukan **detection dan text extraction** pada bukti transfer digital (mobile banking) berupa _nominal, pengirim, penerima, dan tanggal._")
    st.markdown('### ')
    st.info(''' 
    - If you have questions, get in touch with us: dreamrealm369@gmail.com''')
    st.markdown(''' 
    Let's Get Started!
     > 👈🏻 Silakan pilih menu **'Project'** di samping untuk melakukan demo project.
    ''')

video_urls = {'Video 1':'https://www.youtube.com/watch?v=UHX6zmMUShk', 
              'Video 2':'https://www.youtube.com/watch?v=7y4WhR3W2Nc',
              'Video 3':'https://www.youtube.com/watch?v=AclJEg7psPw'}
vid_selection = st.sidebar.selectbox('Pilih video', list(video_urls.keys())) #dropdown unt memilih video
st.sidebar.video(video_urls[vid_selection]) #putar video pilihan
    
# Define the class labels
if (selected == 'Project'):
    class_labels = ['nominal', 'pengirim', 'penerima', 'tanggal']

# --- Load the model ---
#    def load_model():
#        num_classes = 4  # Update the number of classes based on your dataset
#        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
#        in_features = model.roi_heads.box_predictor.cls_score.weight.shape[1]
#        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#        model.load_state_dict(torch.load('trained_model (4).pth', map_location=torch.device('cpu')))
#        model.eval()
#       return model

#    model = load_model()

    # Define the path to pickle file
    model_path = 'trained_model.pkl' #running the code to load the model with cpu (without CUDA support)
    # Load the model from .pkl file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

    #@st.cache(suppress_st_warning=True)
    def prediction(image):
    # Create the OCR reader
        reader = easyocr.Reader(['en'])  # Replace 'en' with the appropriate language code for your OCR needs

    # Preprocess the image
        image_tensor = F.to_tensor(image).unsqueeze(0)

    # Make predictions on the image 
        with torch.no_grad():
            predictions = model(image_tensor)

    # Extract the predicted bounding boxes, labels, and scores
        predicted_bboxes = predictions[0]['boxes'].cpu()
        predicted_labels = predictions[0]['labels'].cpu()
        predicted_scores = predictions[0]['scores'].cpu()

    # Initialize variables to store the highest accuracy bounding box for each label
        label_bboxes = {label_name: None for label_name in class_labels}

    # Find the highest accuracy bounding box for each label
        for bbox, label, score in zip(predicted_bboxes, predicted_labels, predicted_scores):
            label_name = class_labels[label - 1]
            if label_bboxes[label_name] is None or score > label_bboxes[label_name][1]:
                label_bboxes[label_name] = (bbox, score)

    # Extract text within the bounding boxes using OCR
        extracted_text = {}
        for label_name, bbox_score in label_bboxes.items():
            if bbox_score is not None:
                bbox, score = bbox_score
                xmin, ymin, xmax, ymax = bbox.tolist()

            # Crop the region defined by the bounding box
                region = image.crop((xmin, ymin, xmax, ymax))

            # Convert PIL image to NumPy array
                region_np = np.array(region)

            # Perform OCR on the cropped region using easyocr
                result = reader.readtext(region_np)
                if len(result) > 0:
                    text = result[0][1]
                else:
                    text = ""

            # Store the extracted text for the label
                extracted_text[label_name] = text

            # Draw bounding box
                plt.imshow(image)
                ax = plt.gca()
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                label_text = f'{label_name} ({score:.2f}) - {text}'
                plt.text(xmin, ymin, label_text, bbox=dict(facecolor='red', alpha=0.5))
            else:
                extracted_text[label_name] = None

    # Display the image with the bounding boxes and extracted text
        plt.axis('off')
        st.pyplot()

        return extracted_text
 
    # Create the Streamlit app
    
    st.warning("Note: Hanya dapat memproses berupa bukti bank digital", icon="⚠️")

    # Upload image
    uploaded_image = st.file_uploader('Upload your bank transfer invoice here', type=['png', 'jpg', 'jpeg'],
                                  help='Unggah gambar dalam format png, jpg, atau jpeg')

    col1, col2 = st.columns(2)

    #for uploaded_image in uploaded_images:
    if uploaded_image is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_image)
        with col1:
            st.write('Original Image')
            col1.subheader('Uploaded Image')
        #image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', width=350)
            st.success("Image uploaded successfully!", icon="✅")
        result = st.button('Process Image')
        with col2:
            if result:
            #if st.button('Process Image'):
                st.write('Showing results...')
        # Perform prediction and OCR
                st.subheader('Predicted Image')
                extracted_text = prediction(image)

        # Display the extracted text for each label
                st.subheader('Extracted Text')
                for label_name, text in extracted_text.items():
                    st.write(f'{label_name}: {text}')

# Add copyright notice at the bottom of the page
footer_html = '''
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    color: #123;
    text-align: center;
    padding: 10px;
}
</style>

<div class="footer">
    <p>&copy; 2023, Dream Realm.</p>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)
