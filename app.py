import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

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
    
    # Load the weights you trained in Phase 2
    model.load_state_dict(torch.load('saved_models/face_mask_mobilenetv2.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

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

# Prediction Logic
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)
    st.write("Analyzing...")
    
    # Transform image and add batch dimension
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Make Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        
        class_idx = predicted_class.item()
        conf_percent = confidence.item() * 100

    # Display Results
    st.markdown("### Result:")
    if class_idx == 0:
        st.success(f"✅ **{classes[class_idx]}** ({conf_percent:.2f}% confidence)")
    else:
        st.error(f"🚨 **{classes[class_idx]}** ({conf_percent:.2f}% confidence)")