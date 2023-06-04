import os
import cv2
import glob
import easyocr
import matplotlib.pyplot as plt
#import torch
#import torchvision.transforms as T
import xml.dom.minidom
import xml.etree.ElementTree as ET
import pytesseract
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
#from torchvision.models.detection import FasterRCNN
#from torchvision.models.detection.rpn import AnchorGenerator
#from torchvision.transforms import functional as F
from PIL import Image
import pickle
import streamlit as st
from streamlit_option_menu import option_menu as option

pg_title = "Dream Realm Project"
icon = "Logo Dream Realm.png"
email = "dreamrealm369@gmail.com"
description = """(deskripsi)"""

st.set_page_config(page_title=pg_title, 
                   page_icon=icon,  layout="wide") 
st.markdown('<h1 style="text-align: center;">Welcome to Dream Realm</h1>', unsafe_allow_html=True)
st.title("_Object Detection and Text Extraction_")
st.subheader(description)

# Menampilkan Option Menu
with st.sidebar:
        selected = option("Main Menu", ["Home", 'Settings', 'About', 'Contact Us'], 
        icons=['house', 'gear', 'info-circle', 'envelope-at'], menu_icon="list", default_index=1)
        selected
        if selected == 'Home':
            st.write('Anda berada di halaman Home')
        elif selected == 'Settings':
            st.write('Anda berada di halaman Settings')
        elif selected == 'About':
            st.write('Anda berada di halaman About')
        else:
            st.write('Anda berada di halaman Contact us')

video_urls = {'Video 1':'https://www.youtube.com/watch?v=UHX6zmMUShk', 
              'Video 2':'https://www.youtube.com/watch?v=7y4WhR3W2Nc',
              'Video 3':'https://www.youtube.com/watch?v=AclJEg7psPw'}
vid_selection = st.sidebar.selectbox('Pilih video', list(video_urls.keys())) #dropdown unt memilih video
st.sidebar.video(video_urls[vid_selection]) #putar video pilihan
    
def load_image():
    st.warning("Note: Hanya dapat memproses berupa bukti bank digital atau screenshot")
    uploaded_file = st.file_uploader('Upload your bank transfer invoice here', type=['png','jpg','jpeg'], 
                                   help='Unggah gambar dalam format png, jpg, atau jpeg')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue() #read the contents of uploaded file as bytes
        st.image(image_data)
        st.write("filename:", uploaded_file.name)
        st.write(image_data, use_column_width=False)
        image = Image.open(uploaded_file) #membuka gambar
        #resized_gambar = image.resize((10, int(10*image.height/image.width)))
        #st.image(image_data, use_column_width=False) #menampilkan gambar pd streamlit
        st.success("Image uploaded successfully!")
        return image 
    else:
        return None
        #print("")
        #return Image.open(io.BytesIO(image_data))
        #return Image.open(io.BytesIO(image_data))
    #else:
        #return None
        #image = Image.open(uploaded_file)
        # Perform object detection and text extraction
        #extracted_text, output_image = prediction_(uploaded_file)
    
    # Convert the file to an opencv image.
    #file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #opencv_image = cv2.imdecode(file_bytes, 1)
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       # resized = cv2.resize(opencv_image, (224,224))
       # st.image(opencv_image, channels="RGB")
    #save input image
#    try:
 #       image = Image.open(uploaded_file)
  #     st.error("Error: Invalid image")
   # else:
   #     img_array = np.array(image)
   #     return img_array
   # cv2.imwrite('input.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

# Define the class labels
class_labels = ['nominal', 'pengirim', 'penerima', 'tanggal']

# Load the OCR reader
reader = easyocr.Reader(['en'], ['id'])

# --- Load the model ---
def load_model():
    # Define the path to pickle file
    model_path = 'trained_model.pkl' #running the code to load the model with cpu (without CUDA support)
    #model = model_path.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    #model.eval()
    #return model
    #model_path = pickle.load(open('trained_model.pkl', 'rb'))

    # Load the trained model
    # Open the file in binary mode 
    with open(model_path, 'rb') as file:
         model = pickle.load(file)
    model.eval()
    return model_path

# --- Function to make predictions ---
# Define the prediction function
@st.cache(allow_output_mutation=True)
def prediction(model, categories, image):  
    # Preprocess the input data (image)
    new_image_path = image.name 
    new_image = Image.open(new_image_path).convert("RGB")
    new_image_tensor = F.to_tensor(new_image).unsqueeze(0)
    
 #   preprocess = T.Compose([T.ToTensor()])
    # Make predictions using the loaded model
  #  predictions = load_model.predict(image)
   # new_image_path = load_image.__name__
   # new_image = Image.open(new_image_path).convert("RGB")
    #image = Image.open(io.BytesIO(image_bytes)) 
    #new_image_tensor = F.to_tensor(new_image).unsqueeze(0)

    # Make predictions on the new image
    with torch.no_grad():
        predictions = model(new_image_tensor)
    #return predictions

    # Perform object detection and OCR on the uploaded image
    #with st.spinner('Performing object detection and OCR...'):
    #image = load_image()
    #if Image is None:
        #st.warning("Please upload an image")
       #return
    #predictions = prediction(image)

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

# Extracted text dictionary
    extracted_text = {}

# Draw bounding boxes and perform text extraction
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
            draw_rect = st.image(image, caption="Object Detected", use_column_width=True)
            draw_rect.annotate("", bbox=[[xmin, ymin], [xmax, ymax]], annotation_text=f'{label_name} ({score:.2f}) - {text}',
                       annotation_position="outside")
        else:
            extracted_text[label_name] = None

    # Print the extracted text for each label
    st.subheader("Extracted Text")
    for label_name, text in extracted_text.items():
        st.write(f'{label_name}: {text}') # Display the prediction

# --- Streamlit app ---
# Add UI to get user upload data
# Process the upload data and pass it to the predict function
# Display the prediction
def main():
    # Add UI to get user upload data
    model = load_model()
    categories = class_labels
    image = load_image()
    result = st.button('Run on image')
    if result:
       st.write('Show the results...')
       prediction(model, categories, image) # Make predictions

if __name__ == "__main__":
    main()


    # Perform object detection and text extraction
   # st.subheader("Uploaded Image")
   # st.image(image, caption="Uploaded Image", use_column_width=True)
    
      # Now do something with the image! For example, let's display it:
   # print("original Image")
   # st.image(opencv_image, channels="BGR", caption='Original Image')

   # print("Predicted Image")
   # image = Image.open(uploaded_file)
   # st.image(image, caption='Predicted Image')


    # Perform object detection and OCR on the uploaded image
   # with st.spinner('Performing object detection and OCR...'):
    #    predictions = prediction(image)
