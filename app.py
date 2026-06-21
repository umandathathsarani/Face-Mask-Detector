import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# App Configuration
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")
st.title("😷 Real-Time Face Mask Detector")
st.write("Upload an image or take a picture with your webcam to see if you are wearing a mask!")

# Load Model (Cached for speed)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load('saved_models/face_mask_mobilenetv2.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, device, face_cascade

model, device, face_cascade = load_model()

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['With Mask', 'Without Mask']

# UI: Input Options
option = st.radio("Choose an input method:", ("Upload an Image", "Use Webcam"))

image = None

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
elif option == "Use Webcam":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')

# Prediction & Drawing Logic
if image is not None:
    st.write("Analyzing...")
    
    # Convert PIL Image to numpy array (RGB format for Streamlit)
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        st.warning("⚠️ No face detected in this image! Please upload a clear photo of a person.")
        st.image(image, caption="Input Image", use_container_width=True)
    else:
        # Loop through all detected faces
        for (x, y, w, h) in faces:
            face_roi = img_array[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_roi)
            
            input_tensor = preprocess(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)
                
                class_idx = predicted_class.item()
                conf_percent = confidence.item() * 100
            
            # Draw bounding box and label
            # Streamlit uses RGB: Green is (0, 255, 0), Red is (255, 0, 0)
            color = (0, 255, 0) if class_idx == 0 else (255, 0, 0)
            label = f"{classes[class_idx]} ({conf_percent:.1f}%)"
            
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Display the final augmented image
        st.image(img_array, caption="Processed Image", use_container_width=True)
        
        # Summary text below
        if class_idx == 0:
            st.success("Analysis Complete: Mask Detected ✅")
        else:
            st.error("Analysis Complete: No Mask Detected 🚨")