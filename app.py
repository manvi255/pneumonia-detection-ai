# =========================================================
# ADVANCED PROFESSIONAL PNEUMONIA DETECTION UI
# Residual CNN + Grad-CAM + Streamlit
# =========================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

import numpy as np
import cv2

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩻",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 10px;
    }

    .normal-box {
        background-color: #0F9D58;
        color: white;
    }

    .pneumonia-box {
        background-color: #DB4437;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# RESIDUAL BLOCK
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels)

        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),

                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.conv(x)

        out += self.shortcut(x)

        out = self.relu(out)

        return out

# =========================================================
# MAIN CNN
# =========================================================

class PneumoniaCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.layer2 = ResidualBlock(32, 64, stride=2)

        self.layer3 = ResidualBlock(64, 128, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(

            nn.Linear(128, 256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, 2)
        )

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# =========================================================
# LOAD MODEL
# =========================================================

model = PneumoniaCNN().to(device)

checkpoint = torch.load(
   "pneumonia_full_checkpoint_v2.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# =========================================================
# THRESHOLD
# =========================================================

THRESHOLD = 0.80

# =========================================================
# GRAD-CAM VARIABLES
# =========================================================

activations = None
gradients = None

# =========================================================
# HOOKS
# =========================================================

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

model.layer3.register_forward_hook(forward_hook)

model.layer3.register_full_backward_hook(backward_hook)

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5])

])

# =========================================================
# GRAD-CAM FUNCTION
# =========================================================

def generate_gradcam(model, input_tensor, class_idx):

    global activations
    global gradients

    output = model(input_tensor)

    model.zero_grad()

    output[0, class_idx].backward()

    pooled_gradients = torch.mean(
        gradients,
        dim=[0,2,3]
    )

    activations_copy = activations.clone()

    for i in range(activations_copy.shape[1]):

        activations_copy[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(
        activations_copy,
        dim=1
    ).squeeze()

    heatmap = torch.relu(heatmap)

    heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().cpu().numpy()

    return heatmap

# =========================================================
# HEADER
# =========================================================

st.title("🩻 AI Powered Pneumonia Detection System")

st.markdown(
    """
    This deep learning system uses a **Residual CNN + Grad-CAM**
    to detect pneumonia from chest X-ray images.
    """
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Model Information")

st.sidebar.write("Architecture: Residual CNN")
st.sidebar.write("Input Size: 224 × 224")
st.sidebar.write("Threshold: 0.80")
st.sidebar.write("Classes: NORMAL / PNEUMONIA")

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = F.softmax(outputs, dim=1)

        pneumonia_prob = probs[:,1].item()

        normal_prob = probs[:,0].item()

        if pneumonia_prob > THRESHOLD:
            prediction = "PNEUMONIA"
        else:
            prediction = "NORMAL"

    # =====================================================
    # GRAD-CAM
    # =====================================================

    predicted_class = torch.argmax(outputs, dim=1).item()

    heatmap = generate_gradcam(
        model,
        input_tensor,
        predicted_class
    )

    original_image = image.convert("RGB")

    original_np = np.array(original_image)

    original_np = cv2.resize(original_np, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = heatmap * 0.4 + original_np

    superimposed_img = np.clip(
        superimposed_img,
        0,
        255
    ).astype(np.uint8)

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original X-ray")

        st.image(
            original_np,
            use_container_width=True
        )

    with col2:

        st.subheader("Grad-CAM Heatmap")

        st.image(
            superimposed_img,
            use_container_width=True
        )

    # =====================================================
    # PREDICTION BOX
    # =====================================================

    if prediction == "PNEUMONIA":

        st.markdown(
            f'''
            <div class="prediction-box pneumonia-box">
            🦠 Prediction: {prediction}
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'''
            <div class="prediction-box normal-box">
            ✅ Prediction: {prediction}
            </div>
            ''',
            unsafe_allow_html=True
        )

    # =====================================================
    # CONFIDENCE SCORES
    # =====================================================

    st.subheader("Confidence Scores")

    st.write(
        f"Pneumonia Probability: {pneumonia_prob:.4f}"
    )

    st.progress(float(pneumonia_prob))

    st.write(
        f"Normal Probability: {normal_prob:.4f}"
    )

    st.progress(float(normal_prob))

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.subheader("AI Interpretation")

    if prediction == "PNEUMONIA":

        st.warning(
            "The model detected pneumonia-related patterns in the lung region."
        )

    else:

        st.success(
            "The model did not detect significant pneumonia-related abnormalities."
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Deep Learning based Pneumonia Detection using Residual CNN + Grad-CAM"
)