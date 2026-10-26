import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# -------------------------------------------------------------
# 1. Page Configuration & Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Waste Classification & IVP Audit",
    page_icon="♻️️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_finetuned_resnet18.pth"
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# -------------------------------------------------------------
# 2. Grad-CAM Engine (Self-Cleaning Hooks)
# -------------------------------------------------------------
class GradCAM:

  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.activations = None
    for param in self.target_layer.parameters():
      param.requires_grad = True

  def generate_cam(self, input_tensor, class_idx=None):
    self.model.eval()

    def _fwd_hook(module, inp, out):
      self.activations = out.detach()

    def _bwd_hook(module, gin, gout):
      self.gradients = gout[0].detach()

    h_fwd = self.target_layer.register_forward_hook(_fwd_hook)
    h_bwd = self.target_layer.register_full_backward_hook(_bwd_hook)

    try:
      with torch.enable_grad():
        out = self.model(input_tensor)
        if class_idx is None:
          class_idx = torch.argmax(out, dim=1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()

      weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
      cam = torch.sum(weights * self.activations, dim=1).squeeze()
      cam = F.relu(cam)
      cam = cam - cam.min()
      if cam.max() > 0:
        cam = cam / cam.max()
      return cam.cpu().numpy()
    finally:
      h_fwd.remove()
      h_bwd.remove()


def overlay_cam(rgb_img, cam_mask, alpha=0.5):
  heatmap = cv2.resize(cam_mask, (rgb_img.shape[1], rgb_img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
  blended = np.float32(heatmap_colored) * alpha + np.float32(rgb_img) * (
      1 - alpha
  )
  return np.uint8(np.clip(blended, 0, 255))


# -------------------------------------------------------------
# 3. IVP Preprocessing Pipeline
# -------------------------------------------------------------
def run_ivp_pipeline(pil_img):
  img_rgb = np.array(pil_img)
  img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

  gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  enhanced = clahe.apply(blurred)
  edges = cv2.Canny(enhanced, 50, 150)
  _, otsu_mask = cv2.threshold(
      enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  morph_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
  if np.mean(morph_mask == 255) > 0.85:
    morph_mask = cv2.bitwise_not(morph_mask)

  segmented_bgr = cv2.bitwise_and(img_resized, img_resized, mask=morph_mask)
  segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

  return {
      'gray': gray,
      'enhanced': enhanced,
      'edges': edges,
      'mask': morph_mask,
      'segmented': segmented_rgb,
  }


# -------------------------------------------------------------
# 4. Model Loading Cache
# -------------------------------------------------------------
@st.cache_resource
def load_waste_model():
  if not MODEL_PATH.exists():
    return None, None
  checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
  classes = checkpoint['classes']
  print("Checkpoint classes:", classes)

  model = models.resnet18(weights=None)
  num_features = model.fc.in_features
  model.fc = nn.Sequential(
      nn.Dropout(p=0.3), nn.Linear(num_features, len(classes))
  )
  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.to(DEVICE)
  model.eval()
  return model, classes


model, classes = load_waste_model()

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------------------------------------------
# 5. UI Layout & Controls
# -------------------------------------------------------------
st.title("♻️ Automated Solid Waste Classification & Explainability System")
st.markdown(
    "**End-to-End Hybrid Computer Vision System**: Integrating Classical IVP"
    " Preprocessing, Deep Transfer Learning, and Interpretability via Grad-CAM."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Overall Test Accuracy", value="84%")
m2.metric(label="Macro Avg F1-Score", value="0.76")
m3.metric(label="Held-Out Test Samples", value="751 Images")
# m4.metric(label="Inference Device", value="Apple MPS (GPU)")

st.markdown("---")

with st.sidebar:
  st.header("⚙️ Operational Settings")
  confidence_threshold = st.slider(
      "Confidence Guardrail Threshold (%)",
      min_value=50,
      max_value=95,
      value=65,
      step=5,
      help=(
          "Predictions below this confidence percentage trigger an automatic"
          " human inspection flag."
      ),
  )
  st.markdown("---")
  st.subheader("Model Status")
  if model is not None:
    st.success(f"Model Active: ResNet-18 ({DEVICE.type.upper()})")
    st.write(f"Supported Classes: `{len(classes)}`")
    st.info(" **Benchmark Test Acc:** `84%` (751 images)")
  else:
    st.error("Model weights file not found in models/.")

input_source = st.radio(
    "Select Image Input Mode:",
    ("Upload Image File", "Use Web Camera"),
    horizontal=True,
)
uploaded_file = None

if input_source == "Upload Image File":
  uploaded_file = st.file_uploader(
      "Choose a waste image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"]
  )
else:
  camera_image = st.camera_input("Capture item photo")
  if camera_image is not None:
    uploaded_file = camera_image

# -------------------------------------------------------------
# 6. Inference & Dashboard Visualization
# -------------------------------------------------------------
if uploaded_file is not None and model is not None:
  img_raw = Image.open(uploaded_file).convert("RGB")

  # Run IVP Preprocessing
  ivp_data = run_ivp_pipeline(img_raw)

  # Model Prediction
  tensor_img = eval_transform(img_raw).unsqueeze(0).to(DEVICE)
  with torch.no_grad():
    logits = model(tensor_img)
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()


  top_idx = int(np.argmax(probs))
  predicted_class = classes[top_idx]
  confidence = float(probs[top_idx] * 100)

  # Initial uncertainty safeguard
  MIN_CONFIDENCE = 65.0
  is_uncertain = confidence < MIN_CONFIDENCE


  # Waste disposal recommendations
  disposal_guidance = {
      "Organic": {
          "title": "Compost or dispose as organic waste",
          "instructions": [
              "Separate food scraps and suitable garden waste.",
              "Compost them where facilities are available.",
              "Keep plastic, glass, and metal out of the compost.",
          ],
          "warning": "Follow local composting rules. Meat, dairy, and other items may not be accepted by every facility.",
      },
      "Recyclable": {
          "title": "Prepare for recycling",
          "instructions": [
              "Check the material: paper, cardboard, glass, metal, or plastic.",
              "Empty and clean containers when practical.",
              "Keep materials separate according to local recycling rules.",
          ],
          "warning": "Not every item made from a recyclable material is accepted by every local facility.",
      },
      "Non-Recyclable": {
          "title": "Check general waste disposal options",
          "instructions": [
              "Check whether the item can be reused or repaired.",
              "Check local waste collection guidance.",
              "Place it in the designated residual-waste bin if no suitable recovery option exists.",
          ],
          "warning": "This is a broad category. Some items may have specialist collection options.",
      },
      "Hazardous": {
          "title": "Keep separate and arrange safe disposal",
          "instructions": [
              "Do not mix the item with ordinary household waste.",
              "Avoid opening, crushing, burning, or handling leaking contents.",
              "Use an authorized hazardous-waste collection point or facility.",
          ],
          "warning": "The correct procedure depends on the specific item. Follow local guidance, especially for batteries, chemicals, paints, and electronic waste.",
      },
  }

  guidance = disposal_guidance.get(predicted_class)


  # Grad-CAM Heatmap
  grad_cam = GradCAM(model=model, target_layer=model.layer4)
  cam_mask = grad_cam.generate_cam(tensor_img, class_idx=top_idx)
  img_disp = np.array(img_raw.resize((224, 224)))
  cam_overlay = overlay_cam(img_disp, cam_mask)

  st.markdown("---")

# Operational Decision Banner
  is_automated = confidence >= confidence_threshold

  if is_uncertain:
      st.warning(
          f"### ⚠️ UNCERTAIN PREDICTION: {predicted_class.upper()} "
          f"({confidence:.1f}% confidence). Please inspect the image manually."
      )
  elif is_automated:
      st.success(
          f"### ✅ PROVISIONAL CLASSIFICATION: {predicted_class.upper()} "
          f"({confidence:.1f}% confidence)"
      )
  else:
      st.warning(
          f"### ⚠️ MANUAL INSPECTION REQUIRED: {predicted_class.upper()} "
          f"({confidence:.1f}% confidence)"
      )


  # Disposal guidance panel
  if is_uncertain:
      st.info(
          "Disposal advice is withheld because the prediction is uncertain. "
          "Please verify that the image contains waste and identify the item."
      )
  elif guidance:
      st.markdown("### ♻️ Recommended Disposal Action")
      st.subheader(guidance["title"])

      for instruction in guidance["instructions"]:
          st.markdown(f"- {instruction}")

      st.warning(guidance["warning"])

  # Layout: Visual Explanations (Row 1)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.subheader("1. Raw Input")
    st.image(img_raw, use_container_width=True)
  with col2:
    st.subheader("2. IVP Isolated Waste")
    st.image(ivp_data["segmented"], use_container_width=True)
  with col3:
    st.subheader("3. Grad-CAM Saliency")
    st.image(cam_overlay, use_container_width=True)

  # Layout: Analysis & Preprocessing Breakdown (Row 2)
  col_bars, col_ivp = st.columns([1, 1])

  with col_bars:
    st.subheader("Class Probability Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bar_colors = [
        "#2ca02c" if c == predicted_class else "#1f77b4" for c in classes
    ]
    y_pos = np.arange(len(classes))
    ax.barh(
        y_pos,
        probs * 100,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.axvline(
        x=confidence_threshold,
        color="red",
        linestyle="--",
        label=f"Guardrail ({confidence_threshold}%)",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence Score (%)", fontsize=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)

  with col_ivp:
    st.subheader("IVP Processing Stages")
    p1, p2, p3 = st.columns(3)
    with p1:
      st.caption("CLAHE Enhanced")
      st.image(ivp_data["enhanced"], clamp=True, use_container_width=True)
    with p2:
      st.caption("Canny Edges")
      st.image(ivp_data["edges"], clamp=True, use_container_width=True)
    with p3:
      st.caption("Otsu Mask")
      st.image(ivp_data["mask"], clamp=True, use_container_width=True)