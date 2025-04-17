import streamlit as st
import cv2
import numpy as np
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
import base64

# Load model with error handling
try:
    model = load_model('gender_detection.h5', compile=False)
    classes = ['Man', 'Woman']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Camera Module", "About"))

if page == "Home":
    set_background("background.jpeg")
    st.title("Real-Time Gender Detection")
    st.write("This project is a real-time gender detection system that uses deep learning and computer vision to classify gender based on facial features. It processes live webcam footage, detects faces, and predicts gender using a trained model.")
    st.write("APPLICATIONS:")
    st.write("-Can be integrated into AI-powered surveillance systems.")
    st.write("-Useful for demographic analysis and customer insights.")
    st.write("-Enhances interactive applications that adapt based on user identity.")
    st.write("-Showcases the capabilities of machine learning in real-world applications.")

elif page == "Camera Module":
    set_background("cam.jpg")
    st.title("Camera Module")
    
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    start_detection = st.button("Start Detection")
    stop_detection = st.button("Stop Detection")
    frame_placeholder = st.empty()

    if start_detection:
        while webcam.isOpened() and not stop_detection:
            status, frame = webcam.read()
            if not status:
                st.error("Error accessing webcam.")
                break
            
            # Detect faces
            face, confidence = cv.detect_face(frame)
            for idx, f in enumerate(face):
                (startX, startY, endX, endY) = f
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                face_crop = np.copy(frame[startY:endY, startX:endX])
                
                if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue
                
                # Preprocess for model
                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
                
                # Predict gender
                conf = model.predict(face_crop)[0]
                idx = np.argmax(conf)
                label = "{}: {:.2f}%".format(classes[idx], conf[idx] * 100)
                
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            time.sleep(0.05)
        
        webcam.release()
        st.text("Detection stopped.")

elif page == "About":
    set_background("about.jpeg")
    st.title("About")
    st.write("This gender detection model is built using deep learning and OpenCV.")
    st.write("Dataset used: [Insert dataset link here]")
