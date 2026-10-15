"""
Standalone multi-object waste classification frontend.

Run from the project root:
    streamlit run src/multi_object_app.py

This is a separate Streamlit app. It does not modify or extend src/app.py.
It uses YOLOv8n for general-object proposals and the existing ResNet-18
checkpoint to classify each detected crop into the project's waste classes.
"""

from pathlib import Path
import io
import csv

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_expanded_resnet18.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ResNet-18 classifier...")
def load_classifier(model_path_string: str):
    model_path = Path(model_path_string)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Make sure models/best_expanded_resnet18.pth exists."
        )

    try:
        checkpoint = torch.load(
            model_path, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    classes = list(checkpoint["classes"])
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, classes


@st.cache_resource(show_spinner="Loading YOLO object detector...")
def load_detector(model_name: str):
    # The first run may download the YOLO weights.
    return YOLO(model_name)


def classify_crop(model, classes, crop: Image.Image):
    tensor = CLASSIFIER_TRANSFORM(crop.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return classes[index], float(probabilities[index].item())


def load_font():
    for font_path in ("/System/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_path, 18)
        except OSError:
            continue
    return ImageFont.load_default()


def run_multi_object_inference(
    original_image: Image.Image,
    detector,
    classifier,
    classes,
    detector_confidence: float,
    fallback_confidence: float,
    image_size: int,
    classifier_threshold: float,
    max_detections: int,
):
    source_rgb = original_image.convert("RGB")
    source_array = np.array(source_rgb)

    result = detector.predict(
        source=source_array,
        conf=detector_confidence,
        imgsz=image_size,
        max_det=max_detections,
        verbose=False,
    )[0]

    retried = False
    if result.boxes is None or len(result.boxes) == 0:
        retried = True
        result = detector.predict(
            source=source_array,
            conf=fallback_confidence,
            imgsz=image_size,
            max_det=max_detections,
            verbose=False,
        )[0]

    annotated = source_rgb.copy()
    draw = ImageDraw.Draw(annotated)
    font = load_font()
    records = []

    detections = result.boxes
    if detections is not None and len(detections) > 0:
        for index, box in enumerate(detections, start=1):
            coords = box.xyxy[0].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords.tolist()

            x1 = max(0, min(x1, source_rgb.width - 1))
            y1 = max(0, min(y1, source_rgb.height - 1))
            x2 = max(x1 + 1, min(x2, source_rgb.width))
            y2 = max(y1 + 1, min(y2, source_rgb.height))

            # Crop from the original image, never from the annotated image.
            crop = source_rgb.crop((x1, y1, x2, y2))
            predicted_class, class_confidence = classify_crop(
                classifier, classes, crop
            )
            detector_conf = float(box.conf[0].item())
            detector_class = detector.names[int(box.cls[0].item())]
            needs_review = class_confidence < classifier_threshold

            if needs_review:
                label = f"REVIEW: {predicted_class} {class_confidence * 100:.0f}%"
                outline = (220, 120, 0)
            else:
                label = f"{predicted_class} {class_confidence * 100:.0f}%"
                outline = (0, 150, 0)

            draw.rectangle((x1, y1, x2, y2), outline=outline, width=4)
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            label_top = max(0, top)
            draw.rectangle(
                (left, label_top, right + 6, bottom + 4), fill=outline
            )
            draw.text((x1 + 3, label_top + 2), label, fill="white", font=font)

            records.append({
                "Object #": index,
                "YOLO label": detector_class,
                "YOLO confidence (%)": round(detector_conf * 100, 2),
                "Waste classification": predicted_class,
                "Classifier confidence (%)": round(class_confidence * 100, 2),
                "Review required": "Yes" if needs_review else "No",
                "Bounding box (x1, y1, x2, y2)": f"{x1}, {y1}, {x2}, {y2}",
            })

    return annotated, records, retried


# ---------------------------------------------------------------------
# Standalone Streamlit page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Object Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("♻️ Multi-Object Waste Classification")
st.caption(
    "Detect multiple objects in one image, classify each detected crop, "
    "and flag uncertain predictions for manual review."
)

st.info(
    "Detection uses a pretrained general-purpose YOLO model, not a waste-specific "
    "detector. It can miss waste items or propose boxes around unrelated objects. "
    "The ResNet-18 classifier then predicts a waste category for each proposed crop."
)

with st.sidebar:
    st.header("⚙️ Detection settings")
    yolo_model_name = st.selectbox(
        "YOLO model",
        options=["yolov8n.pt"],
        index=0,
        help="The model weights may be downloaded automatically on first use.",
    )
    detector_confidence = st.slider(
        "YOLO detection confidence",
        min_value=0.05,
        max_value=0.75,
        value=0.10,
        step=0.05,
        help="Lower values may detect more objects but can increase false positives.",
    )
    fallback_confidence = st.slider(
        "Retry confidence if no objects are found",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )
    classifier_threshold = st.slider(
        "Classifier review threshold (%)",
        min_value=40,
        max_value=95,
        value=65,
        step=5,
        help="Predictions below this confidence are flagged for review.",
    )
    image_size = st.select_slider(
        "Detection image size",
        options=[640, 800, 960, 1280],
        value=1280,
    )
    max_detections = st.slider(
        "Maximum objects per image",
        min_value=10,
        max_value=100,
        value=100,
        step=10,
    )
    st.markdown("---")
    st.write(f"**Inference device:** `{DEVICE}`")
    st.write(f"**Classifier checkpoint:** `{MODEL_PATH.name}`")

left, right = st.columns([1, 1])
with left:
    uploaded_file = st.file_uploader(
        "Upload an image containing multiple items",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Use a clear photo with visible objects for the best results.",
    )
with right:
    st.markdown("#### What you will get")
    st.markdown(
        "- Annotated image with a box around each detected object\n"
        "- Waste class and classifier confidence for each box\n"
        "- Human-review flags for low-confidence predictions\n"
        "- Downloadable CSV report"
    )

if uploaded_file is None:
    st.markdown("---")
    st.subheader("How it works")
    step1, step2, step3 = st.columns(3)
    step1.markdown("**1. Detect**\n\nYOLO proposes object bounding boxes.")
    step2.markdown("**2. Classify**\n\nResNet-18 classifies each crop.")
    step3.markdown("**3. Review**\n\nLow-confidence predictions are flagged.")
else:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open this image: {exc}")
        st.stop()

    st.image(original_image, caption="Uploaded image", use_container_width=True)

    run_button = st.button(
        "🔍 Detect and classify objects",
        type="primary",
        use_container_width=True,
    )

    if run_button:
        try:
            with st.spinner("Loading models and analysing the image..."):
                classifier, classes = load_classifier(str(MODEL_PATH))
                detector = load_detector(yolo_model_name)
                annotated, records, retried = run_multi_object_inference(
                    original_image=original_image,
                    detector=detector,
                    classifier=classifier,
                    classes=classes,
                    detector_confidence=detector_confidence,
                    fallback_confidence=fallback_confidence,
                    image_size=image_size,
                    classifier_threshold=classifier_threshold / 100.0,
                    max_detections=max_detections,
                )

            st.session_state["multi_object_result"] = {
                "annotated": annotated,
                "records": records,
                "source_name": uploaded_file.name,
                "retried": retried,
            }
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

    result_data = st.session_state.get("multi_object_result")
    if result_data and result_data.get("source_name") == uploaded_file.name:
        st.markdown("---")
        st.subheader("Analysis results")

        records = result_data["records"]
        annotated = result_data["annotated"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects detected", len(records))
        m2.metric(
            "Flagged for review",
            sum(row["Review required"] == "Yes" for row in records),
        )
        m3.metric(
            "Inference device",
            DEVICE.type.upper(),
        )

        if result_data["retried"]:
            st.warning(
                "The first detection pass found no objects, so a second pass "
                "used the lower fallback confidence. This may add false positives."
            )

        image_col, details_col = st.columns([1.25, 1])
        with image_col:
            st.markdown("#### Annotated image")
            st.image(annotated, use_container_width=True)
            image_buffer = io.BytesIO()
            annotated.save(image_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=image_buffer.getvalue(),
                file_name="multi_object_annotated.png",
                mime="image/png",
                use_container_width=True,
            )

        with details_col:
            st.markdown("#### Per-object results")
            if records:
                df = pd.DataFrame(records)
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results CSV",
                    data=csv_bytes,
                    file_name="multi_object_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.warning(
                    "No objects were detected. Try a clearer image, reduce the "
                    "YOLO confidence, or use an image where items are separated."
                )

        st.caption(
            "YOLO confidence measures confidence in the proposed object box. "
            "Classifier confidence measures confidence in the waste category; "
            "they are different scores. Treat predictions as assistance, not "
            "a guarantee of correct disposal."
        )
