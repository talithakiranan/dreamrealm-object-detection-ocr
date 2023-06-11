import easyocr
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu as option
import requests

pg_title = "Dream Realm Project"
icon = "logo/Dream_Realm.png"

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
    selected = option("Menu", ['Home',  'Project'], 
    icons=['house', 'gear'], menu_icon="list", default_index=0, 
        styles={"container": {"padding": "1!important", "background-color": "#eecef7"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#d6a0a0"},})

if (selected == 'Home'):
    #st.title("_Object Detection and Text Extraction_")

    col1, col2 = st.columns([7,4], gap="small")
    with st.container():
        with col1: 
            st.markdown('''
            <h1 style="text-align: center;"> Welcome to Dream Realm</h1>
            <p style="text-align: center; font-weight: bold; font-size: 22px;"> This is our first project about Object Detection and Text Extraction. </p>
            <p style="text-align: center; "font-weight:italic; font-size: 22px;"> Get to know more about us, read section below </p> ''',
            unsafe_allow_html=True)
    
        with col2:
            st.image(icon, width=300)

    st.write('---')    
    st.markdown(''' 
                **Let's Get Started!**
                > 👈🏻 Silakan pilih menu **'Project'** di samping untuk melakukan demo project.
                ''')
    
    with st.expander("📌 Read me!"):
    # Use html 
        st.subheader("📍 About Us")
        kolom1, kolom2 = st.columns(2)
        with kolom1:
            st.markdown('''
                <h4 style="font-weight= bold;"> Deskripsi Project </h4>
                <p>
                Dream Realm adalah team yang memiliki sebuah project Computer Vision dengan lingkup Object Detection dan OCR. 
                Pada project ini pengguna dapat melakukan detection dan text extraction pada bukti transfer digital (mobile banking) berupa nominal, pengirim, penerima, dan tanggal.
                User mengupload sebuah gambar bukti transfer mobile banking, gambar dapat berupa gambar asli atau screenshot bukti transfer. </p> ''',
                unsafe_allow_html=True)
        with kolom2:
            st.markdown('''
                <h4 style="font-weight= bold;"> Kegunaan Aplikasi </h4>
                <p>
                Aplikasi ini dapat membantu mempermudah user dalam merekap administrasi bukti transfer mobile banking.
                Aplikasi ini open-source dan free untuk digunakan. Terima kasih sudah mengunjungi! </p> ''',
                unsafe_allow_html=True)
    
        st.write('---')
        st.subheader("🔎 Panduan Pengguna")
        st.markdown('''
            **Cara penggunaan aplikasi:**
            1. Pilih menu 'Project' pada sidebar di sisi kiri
            2. Upload gambar bukti transfer mobile banking yang ingin diprediksi
            3. Tunggu hingga hasil upload gambar muncul
            4. Klik button 'predict image', ini memerlukan waktu sampai hasil prediksi keluar
            5. Setelah hasil muncul, harap cek kembali hasil prediksi
            ''')
        st.warning("Hasil prediksi mungkin tidak 100% akurat benar. Project berada dalam tahap pengembangan, diharapkan user dapat memberikan kritik atau saran untuk pengembangan yang lebih baik. Harap bijak dalam menggunakan aplikasi.")
        st.info("Catatan: Jika memiliki saran, kritik atau pertanyaan, kami merekomendasikan untuk mengirim pesan pada bagian 'Contact'")
        
        st.write('---')
        st.subheader("📧 Contact")
        st.info(''' 
            - If you have questions, get in touch with us: dreamrealm369@gmail.com''') 
        st.write("**Atau beri saran, kritik, pertanyaan Anda disini**")
        with st.form(key='form1'):
            name = st.text_input("Nama Anda")
            email = st.text_input("Email Anda")
            pesan = st.text_input("Tulis Pesan Anda")
            button = st.form_submit_button("Kirim")
        if button:
            st.success("Pesan Anda telah terkirim!".format(name))
    
if (selected == 'Project'):
    # Define the class labels
    
    class_labels = ['nominal', 'pengirim', 'penerima', 'tanggal']
    
    def load_model():
        num_classes = 4  # Update the number of classes based on your dataset
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.weight.shape[1]
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model_loc = 'https://drive.google.com/uc?export=download&id=1wXWqUpCaqWREaqMi3Lft3LE1K-ngECGp'
        response = requests.get(model_loc)

        if response.status_code == 200:
            file_content = response.content
            
        else:
            st.write('Error:', response.status_code)
            st.stop()
        
        model_path = './trained_model (7).pth' 
        with open (model_path, 'wb') as file:
                file.write(file_content)
                st.write('File saved successfully!')

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def prediction(model, image):
        # Create the OCR reader
        reader = easyocr.Reader(["en"])  # Replace 'en' with the appropriate language code for your OCR needs

        # Preprocess the image
        image_tensor = F.to_tensor(image).unsqueeze(0)

        # Make predictions on the image
        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract the predicted bounding boxes, labels, and scores
        predicted_bboxes = predictions[0]["boxes"].cpu()
        predicted_labels = predictions[0]["labels"].cpu()
        predicted_scores = predictions[0]["scores"].cpu()

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
                rect = plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                    )
                ax.add_patch(rect)
                label_text = f"{label_name} ({score:.2f}) - {text}"
                plt.text(
                    xmin, ymin, label_text, bbox=dict(facecolor="red", alpha=0.5)
                )
            else:
                extracted_text[label_name] = None

    # Display the image with the bounding boxes and extracted text
        plt.axis("off")
        st.pyplot()

        return extracted_text

    # Create the Streamlit app
    st.warning(
        "Note: Hanya dapat memproses berupa bukti bank digital", icon="⚠️"
    )

    # Upload image
    uploaded_image = st.file_uploader(
        "Upload your bank transfer invoice here",
        type=["png", "jpg", "jpeg"],
        help="Unggah gambar dalam format png, jpg, atau jpeg",
    )

    col1, col2 = st.columns(2)

    #for uploaded_image in uploaded_images:
    if uploaded_image is not None:
    # Read and display the uploaded image
        image = Image.open(uploaded_image)
        with col1:
            st.write("Original Image")
            col1.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Image", width=350)
            st.success("Image uploaded successfully!", icon="✅")
        result = st.button("Process Image")
        with col2:
            if result:
                # if st.button('Process Image'):
                st.write("Showing results...")
                # Perform prediction and OCR
                st.subheader("Predicted Image")
                extracted_text = prediction(model, image)

                # Display the extracted text for each label
                st.subheader("Extracted Text")
                for label_name, text in extracted_text.items():
                    st.write(f"{label_name}: {text}")

# Add copyright at the bottom of the page
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
